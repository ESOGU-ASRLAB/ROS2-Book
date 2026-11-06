#!/usr/bin/env python3
"""
Kapı görevi: çok robotlu algılama / temizleme operatör paneli (Bölüm 8).

Flask + SocketIO arka ucu; HARMONY senaryosunun operatör panelinden türetilmiştir.

HARMONY PANELİNDEN FARKI
────────────────────────
Bu tarafta neredeyse hiçbir fark yoktur ve bu bilerek böyledir. Panelin konuştuğu her
konu -- /harmony/cmd_input, /harmony/robot_status, kusur konuları,
/harmony/scenario_event -- doors_mission_node tarafından aynen uygulanır; bu yüzden
grafikler, kusur tablosu ve açılır pencereler değişmeden çalışır.

Gerçek farklar:

  * MISSION_CMD, ch08_door_inspection paketinin doors_mission.launch.py dosyasını
    başlatır. Bu başlatma doors_mission_node'u (kapı muayenesini o yönetir) ve
    HARMONY senaryosunun kullandığı cleaning_mission_runner'ın AYNISINI çalıştırır.
  * İki ek anahtar -- only_sim ve force_replan -- o başlatmaya iletilir.
  * İKİ kol da çizilir: UR'nin 7 eklem serisi yerine 14 seri.
  * Yan yana iki kamera paneli: gerçek hücre ve Gazebo ikizi.
  * KPI3 senaryo arızası YOKTUR; o, HARMONY doğrulama koşusuna özgüdür.
  * Port 8082; böylece bu panel ile HARMONY paneli yan yana çalışabilir.

Algılamanın kendisi gerçekten farklıdır: HARMONY yalnızca UR ile, elle yazılmış ara
noktalar üzerinden tarıyordu; burada iki kol planlanmış kaplama turunu koşar ve tur
saniyeler değil dakikalar sürer. robot_status'un "Scanning: UR 4/9 Kawasaki 2/3"
biçiminde ilerleme taşımasının sebebi budur; doors_mission_node bunu başlatma
çıktısından okur.

Dört kusur aynı dört kusurdur; görev düğümü onları yeniden tanımlamaz,
book_pymoveit2.harmony_defects'ten içe aktarır. Böylece iki senaryo birbirinden
ayrışamaz.

Panel alt süreçleri başlatmadan önce ROS 2 Humble'ı ve kitabın çalışma alanını
kaynaklar. Çalışma alanı varsayılan olarak ~/ros2_ws'tir (ana README'deki kurulum);
farklıysa ROS2_WS ortam değişkeniyle verilir.
"""

import os
import signal
import subprocess
import threading
import time
import json
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, Iterator, List, Optional, Sequence

from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO, emit

# ==============================================================================
# Çalışma alanı ve çıktı yolları
# ==============================================================================
# Kitabın çalışma alanı; ROS2_WS ortam değişkeniyle değiştirilebilir.
ROS2_WS = os.path.expanduser(os.environ.get("ROS2_WS", "~/ros2_ws"))
# Her alt süreç komutundan önce çalıştırılan ortam kaynaklama satırı
SOURCE_PREFIX = ("source /opt/ros/humble/setup.bash && "
                 f"source {os.path.join(ROS2_WS, 'install', 'setup.bash')}")
# Panel günlükleri kod klasörüne değil kullanıcının çıktı klasörüne yazılır
LOG_DIR = os.path.join(os.path.expanduser("~"), "ros2_book_output", "ch08", "ui_logs")

# ==============================================================================
# Flask uygulamasının kurulumu
# ==============================================================================
app = Flask(__name__,
            template_folder="templates",
            static_folder="static")
app.config["SECRET_KEY"] = "doors-sensing-cleaning-2026"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ==============================================================================
# HARMONY doğrulama senaryosu sabitleri
# ==============================================================================

# Kusur konuları (book_pymoveit2.harmony_defects ile aynı adlar)
TOPIC_DEFECT = "/harmony/mock_perception/defect"
TOPIC_DEFECT_LIST = "/harmony/mock_perception/defect_list"
TOPIC_DEFECT_STATUS = "/harmony/defect_status"

# NOT: HARMONY paneli burada bir de KPI3 senaryo arızası tanımlar ("E-1042,
# Lineer eksen haberleşmesi sağlanamıyor"); algılama son bakış noktasına vardığında
# tetiklenir. HARMONY doğrulama koşusuna aittir; kapı panelinden üreticisi, soket
# işleyicileri ve penceresiyle birlikte çıkarılmıştır.

# ==============================================================================
# Genel durum
# ==============================================================================


class ProcessManager:
    """Kapı görevinin ROS 2 başlatma süreçlerinin yaşam döngüsünü yönetir."""

    HIL_CMD = "ros2 launch my_robot_cell_control hil_test_whole_unified.launch.py harmony:=true"
    # KAPILAR: doors_mission_node + değiştirilmemiş temizleme yürütücüsü. sensing_robot.py
    # bu başlatmada BİLEREK yoktur: aynı /harmony/cmd_input'a abone olur ve START'ta
    # kendi taramasını başlatırdı; UR'ye iki yerden komut gitmiş olurdu.
    MISSION_CMD = "ros2 launch ch08_door_inspection doors_mission.launch.py"

    def __init__(self, socketio_instance: SocketIO) -> None:
        self.socketio = socketio_instance
        self.hil_process = None
        self.mission_process = None
        # Durum değerleri: stopped, starting, running, stopping
        # (durdu, başlıyor, çalışıyor, duruyor)
        self.hil_status = "stopped"
        self.mission_status = "stopped"
        self.robot_confirmed = False
        self._lock = threading.Lock()
        self._log_threads = []

        # Günlük dosyası: kod klasörüne değil ~/ros2_book_output/ch08/ui_logs altına yazılır
        log_dir = LOG_DIR
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filepath = os.path.join(log_dir, f"doors_log_{timestamp}.txt")
        print(f"[Log] Logs will be written to: {self.log_filepath}")

    def _emit_status(self) -> None:
        """Güncel durumu bağlı tüm istemcilere gönderir."""
        self.socketio.emit("status_update", {
            "hil_status": self.hil_status,
            "mission_status": self.mission_status,
            "robot_confirmed": self.robot_confirmed,
        })

    def _emit_log(self, source: str, message: str) -> None:
        """Günlük iletisini bağlı tüm istemcilere gönderir ve dosyaya yazar."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.socketio.emit("log_message", {
            "timestamp": timestamp,
            "source": source,
            "message": message,
        })
        try:
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] [{source}] {message}\n")
        except Exception:
            pass

    def _stream_output(self, process: subprocess.Popen, source_name: str) -> None:
        """Alt sürecin stdout/stderr çıktısını arayüzün günlük paneline aktarır."""
        try:
            for line in iter(process.stdout.readline, ""):
                if line:
                    self._emit_log(source_name, line.rstrip())
                if process.poll() is not None:
                    break
        except (ValueError, OSError):
            pass

    def _kill_process(
            self, process: Optional[subprocess.Popen], name: str, timeout: float = 10) -> None:
        """Süreci kademeli sonlandırır: SIGINT → bekle → SIGTERM → bekle → SIGKILL."""
        if process is None or process.poll() is not None:
            return

        self._emit_log("SYSTEM", f"🛑 Stopping {name} (SIGINT)...")

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except (ProcessLookupError, OSError):
            return

        try:
            process.wait(timeout=timeout)
            self._emit_log("SYSTEM", f"✅ {name} successfully stopped.")
            return
        except subprocess.TimeoutExpired:
            pass

        self._emit_log("SYSTEM", f"⚠️ {name} did not respond, sending SIGTERM...")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
            self._emit_log("SYSTEM", f"✅ {name} stopped via SIGTERM.")
            return
        except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
            pass

        self._emit_log("SYSTEM", f"🔴 Force killing {name} (SIGKILL)...")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=3)
        except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
            pass

        self._emit_log("SYSTEM", f"✅ {name} closed.")

    def _start_process(self, cmd: str, name: str) -> subprocess.Popen:
        """Temiz kapanış için alt süreci kendi süreç grubunda başlatır."""
        self._emit_log("SYSTEM", f"🚀 Starting: {cmd}")

        full_cmd = f"{SOURCE_PREFIX} && {cmd}"

        my_env = os.environ.copy()
        if "QT_QPA_PLATFORM_PLUGIN_PATH" in my_env:
            del my_env["QT_QPA_PLATFORM_PLUGIN_PATH"]

        process = subprocess.Popen(
            full_cmd,
            shell=True,
            executable="/bin/bash",
            env=my_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )

        log_thread = threading.Thread(
            target=self._stream_output,
            args=(process, name),
            daemon=True,
        )
        log_thread.start()
        self._log_threads.append(log_thread)

        return process

    def start_hil(self, use_fake_hardware: bool = False) -> None:
        """HIL sistemini başlatır."""
        def _run() -> None:
            with self._lock:
                if self.hil_process:
                    self._emit_log("SYSTEM", "📋 Stopping current process...")
                    self.hil_status = "stopping"
                    self._emit_status()
                    self._kill_process(self.hil_process, "HIL")
                    self.hil_process = None
                    self.hil_status = "stopped"
                    self._emit_status()
                    time.sleep(2)

                self.hil_status = "starting"
                self._emit_status()

                hil_cmd = self.HIL_CMD
                if use_fake_hardware:
                    hil_cmd += (" use_fake_hardware:=true use_mock_hardware:=true"
                                " fake_sensor_commands:=true")

                self.hil_process = self._start_process(hil_cmd, "HIL")
                self.hil_status = "running"
                self.robot_confirmed = False
                self._emit_status()

                self._emit_log("SYSTEM",
                               "✅ HIL system initialized. "
                               "Please confirm when robots are ready.")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def confirm_robot_ready(self, only_sim: bool = True, force_replan: bool = False) -> None:
        """
        Kullanıcı robotların hazır olduğunu onaylar → görev başlatması çalıştırılır.

        only_sim / force_replan kapı işine özgüdür ve doors_mission.launch.py'ye iletilir;
        o da START geldiğinde bunları muayene başlatmasına verir. HIL başlangıcında değil
        CONFIRM anında okunurlar; böylece operatör hücre ayağa kalkarken onları hâlâ
        değiştirebilir.
        """
        def _run() -> None:
            with self._lock:
                if self.hil_status != "running":
                    self._emit_log("SYSTEM", "❌ HIL is not running, cannot confirm.")
                    return

                self.robot_confirmed = True
                self.mission_status = "starting"
                self._emit_status()

                mission_cmd = (
                    f"{self.MISSION_CMD}"
                    f" only_sim:={'true' if only_sim else 'false'}"
                    f" force_replan:={'true' if force_replan else 'false'}"
                )
                self._emit_log("SYSTEM", "✅ Robot confirmation received. Starting mission...")
                if force_replan:
                    self._emit_log(
                        "SYSTEM",
                        "♻️ force_replan is ON: the cache is ignored and every viewpoint "
                        "is planned again. Slower, but required after a collision-model "
                        "change (cached paths are replayed WITHOUT a collision recheck).")

                self.mission_process = self._start_process(
                    mission_cmd, "MISSION"
                )
                self.mission_status = "running"
                self._emit_status()

                self._emit_log("SYSTEM", "🚀 Sensing & Cleaning Mission is running!")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def stop_all(self) -> None:
        """Acil durdurma: her şeyi sonlandırır."""
        with self._lock:
            self._emit_log("SYSTEM", "🚨 EMERGENCY STOP — Terminating all processes!")

            self.mission_status = "stopping"
            self._emit_status()
            self._kill_process(self.mission_process, "Mission")
            self.mission_process = None
            self.mission_status = "stopped"

            self.hil_status = "stopping"
            self._emit_status()
            self._kill_process(self.hil_process, "HIL")
            self.hil_process = None
            self.hil_status = "stopped"

            self.robot_confirmed = False

            self._emit_status()
            self._emit_log("SYSTEM", "✅ All processes stopped.")

    def get_status(self) -> Dict[str, Any]:
        """Güncel durumu sözlük olarak döndürür."""
        return {
            "hil_status": self.hil_status,
            "mission_status": self.mission_status,
            "robot_confirmed": self.robot_confirmed,
        }


process_mgr = ProcessManager(socketio)


# ==============================================================================
# ROS 2 veri toplayıcı (eklem durumları, robot durumu, kuvvet, kusur)
# ==============================================================================


class ROS2DataCollector:
    """ROS 2 konularından veri toplar ve SocketIO ile arayüze iletir."""

    def __init__(self, socketio_instance: SocketIO, buffer_seconds: int = 20) -> None:
        self.socketio = socketio_instance
        self.buffer_seconds = buffer_seconds

        # Eklem durumu tamponları. Konum + hız değil İKİ KONUM AKIŞI: iki eklem grafiği bir
        # HIL karşılaştırmasıdır -- solda gerçek hücre, sağda Gazebo ikizi; eklemler ve
        # renkler aynıdır, böylece ikisi arasındaki ayrışma bir bakışta görülür. Hızlar
        # artık toplanmıyor; ikinci grafik benzetim aynası olunca onları kullanan bir şey
        # kalmadı.
        self.joint_position_data = deque(maxlen=buffer_seconds * 1000)
        self.sim_position_data = deque(maxlen=buffer_seconds * 1000)

        # Kuvvet verisi tamponu
        self.force_data = deque(maxlen=buffer_seconds * 1000)

        # Robot durumu
        self.current_mode = "IDLE"
        self.current_note = ""

        # Kusur tablosu: defect_id -> {id, type, x, y, z, status, frame_id}
        # Sıralama algılamanın yayın sırasına göre korunur (DEF-01 ... DEF-04).
        self.defects = {}
        self.defect_order = []
        # "Algılama tamamlandı" penceresi tur başına bir kez gösterilir.
        self._sensing_complete_notified = False
        # "Temizleme tamamlandı" penceresi tur başına bir kez gösterilir.
        self._cleaning_complete_notified = False

        # Kalıcı komut yayıncısı. `ros2 topic pub -1` her çağrıda yeni bir düğüm kurup
        # keşif (discovery) tamamlanmadan çıktığı için ilk tıklamalar kaybolabiliyordu;
        # burada uzun ömürlü düğüm üzerinden yayımlıyoruz.
        self._cmd_pub = None

        self._rclpy_thread = None
        self._running = False
        self._data_lock = threading.Lock()

    def publish_cmd(self, cmd_name: str) -> bool:
        """
        Komutu kalıcı ROS düğümü üzerinden yayımlar.

        Yayımlandıysa True döndürür; ROS kullanılamıyorsa False döner (çağıran taraf
        alt süreç yayıncısına geçer).
        """
        pub = self._cmd_pub
        if pub is None:
            return False
        try:
            from std_msgs.msg import String
            msg = String()
            msg.data = json.dumps({"cmd": cmd_name})
            pub.publish(msg)
            return True
        except Exception as e:
            print(f"[ROS2DataCollector] publish_cmd error: {e}")
            return False

    # ------------------------------------------------------------------
    def _upsert_defect(self, payload: Dict[str, Any]) -> None:
        """Gelen kusur JSON'unu tabloya ekler ya da günceller."""
        defect_id = str(payload.get("defect_id", "")).strip()
        if not defect_id:
            return
        pos = payload.get("position", {}) or {}
        entry = {
            "id": defect_id,
            "type": payload.get("defect_type", ""),
            "x": float(pos.get("x", 0.0)),
            "y": float(pos.get("y", 0.0)),
            "z": float(pos.get("z", 0.0)),
            "frame_id": payload.get("frame_id", "world"),
            "status": payload.get("status", "DETECTED"),
        }
        with self._data_lock:
            if defect_id not in self.defects:
                self.defect_order.append(defect_id)
            else:
                # Durum bilgisi ayrı konudan gelmiş olabilir; ezme.
                entry["status"] = self.defects[defect_id].get("status", entry["status"])
            self.defects[defect_id] = entry

    def _defect_snapshot(self) -> List[Dict[str, Any]]:
        with self._data_lock:
            return [self.defects[d] for d in self.defect_order if d in self.defects]

    def clear_defects(self) -> None:
        """Yeni bir algılama turu başlarken eski kusurları temizler."""
        with self._data_lock:
            self.defects = {}
            self.defect_order = []
            self._sensing_complete_notified = False
            self._cleaning_complete_notified = False
        self.socketio.emit("defect_list", {"defects": []})

    def reset_ui(self) -> None:
        """
        Arayüzü ilk görev başlangıcındaki gibi sıfırlar.

        Kusur tablosu, sayaçlar, kip göstergesi ve pencere bayrakları temizlenir; arayüz
        grafik/günlük tarafını da sıfırlasın diye 'mission_reset' yayımlanır. Robot
        süreçlerine (HIL/görev) dokunmaz; yalnızca gösterim durumunu sıfırlar.
        """
        self.clear_defects()  # kusurlar + bayraklar + boş defect_list
        with self._data_lock:
            self.current_mode = "IDLE"
            self.current_note = ""
        self.socketio.emit("robot_info", {"mode": "IDLE", "note": "", "defect_count": 0})
        self.socketio.emit("mission_reset", {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })

    def _notify_sensing_complete(self, count: int) -> None:
        """Algılama tamamlandığında operatöre pencere gönderir (tur başına bir kez)."""
        with self._data_lock:
            if self._sensing_complete_notified:
                return
            self._sensing_complete_notified = True
        self.socketio.emit("sensing_complete", {
            "count": count,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })

    def _notify_cleaning_complete(self, data: Dict[str, Any]) -> None:
        """Temizleme tamamlandığında operatöre pencere gönderir (tur başına bir kez)."""
        with self._data_lock:
            if self._cleaning_complete_notified:
                return
            self._cleaning_complete_notified = True
        try:
            cleaned = int(data.get("cleaned", 0))
        except (TypeError, ValueError):
            cleaned = 0
        try:
            total = int(data.get("total", 0))
        except (TypeError, ValueError):
            total = 0
        self.socketio.emit("cleaning_complete", {
            "cleaned": cleaned,
            "total": total,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })

    def start(self) -> None:
        """ROS 2 abone iş parçacığını başlatır."""
        self._running = True
        self._rclpy_thread = threading.Thread(target=self._ros_spin, daemon=True)
        self._rclpy_thread.start()

        self._push_thread = threading.Thread(target=self._periodic_push, daemon=True)
        self._push_thread.start()

    def _ros_spin(self) -> None:
        """ROS 2 abone döngüsü (spin)."""
        try:
            import rclpy
            from rclpy.node import Node as RosNode
            from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy,
                                   QoSProfile, QoSReliabilityPolicy)
            from sensor_msgs.msg import JointState
            from std_msgs.msg import String
            from geometry_msgs.msg import WrenchStamped

            rclpy.init()
            node = RosNode("doors_dashboard_listener")

            # defect_list kalıcı (latched) yayımlanır; geç bağlanan arayüz de listeyi alır.
            latched = QoSProfile(
                depth=1,
                history=QoSHistoryPolicy.KEEP_LAST,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            )

            # İKİ kol. Kapı işi iki kollu bir turdur; bu yüzden grafikler UR'nin yanında
            # Kawasaki'yi de gösterir (HARMONY panelinde yalnızca UR yetiyordu, çünkü
            # algılama yalnızca o kolda koşuyordu).
            #
            # Aşağıdaki adlar ORTAK (kanonik) adlardır. Hem gerçek hücre hem Gazebo ikizi
            # _canonical_joint() ile bunlara eşlenir; böylece aynı eklem iki grafikte de aynı
            # anahtarı, rengi ve çizgi biçimini korur -- gerçek/benzetim karşılaştırmasını
            # okunabilir kılan tek yol budur. Çalışan hücrede doğrulandı:
            #
            #   /joint_states      ur10e_shoulder_pan_joint  joint1 .. joint6  world_to_agv
            #   /sim/joint_states  sim_ur10e_shoulder_pan_joint  sim_kawasaki_joint1 ..
            #                      sim_world_to_agv
            #
            # Gerçek UR rayı ön ekli olarak ur10e_base_to_robot_mount adıyla gelir; iki konu
            # wheel_left_base / wheel_right_base eklemlerini de taşır; onlar bu listede
            # olmadığı için atılır.
            self.target_joints = [
                # UR10e + doğrusal rayı
                "shoulder_pan_joint", "shoulder_lift_joint",
                "elbow_joint", "wrist_1_joint",
                "wrist_2_joint", "wrist_3_joint",
                "base_to_robot_mount",
                # Kawasaki RS005L + AGV rayı
                "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
                "world_to_agv",
            ]

            def _canonical_joint(name: str) -> str:
                """
                Gerçek YA DA benzetim eklem adını ortak (kanonik) ada eşler.

                Sıra önemlidir: sim_ur10e_ ve sim_kawasaki_ ön ekleri, yalın sim_ ön ekinden önce
                denenmelidir; yoksa sim_ur10e_elbow_joint, elbow_joint yerine ur10e_elbow_joint
                olarak çıkar.
                """
                if name.startswith("sim_ur10e_"):
                    return name[len("sim_ur10e_"):]
                if name.startswith("sim_kawasaki_"):
                    return name[len("sim_kawasaki_"):]
                if name.startswith("sim_"):
                    return name[len("sim_"):]  # sim_world_to_agv -> world_to_agv
                if name.startswith("ur10e_"):
                    return name[len("ur10e_"):]
                return name  # joint1..6, world_to_agv

            def _collect_positions(msg: JointState, buffer: Deque[Dict[str, float]]) -> None:
                """
                Konumları zip ile değil sıra numarasıyla eşler.

                zip(name, position) en kısa dizide durur; kısa dizi gönderen bir yayıncı eklemleri
                sessizce düşürürdü. Kol yayıncıları ile AGV rayı köprüsü burada AYRI iletiler
                yayımlar; yani her ileti meşru olarak hücrenin yalnızca bir kısmını taşır.
                """
                data = {"t": time.time()}
                positions = msg.position or []
                for i, name in enumerate(msg.name):
                    canonical = _canonical_joint(name)
                    if canonical in self.target_joints and i < len(positions):
                        data[canonical] = positions[i]
                if len(data) > 1:
                    with self._data_lock:
                        buffer.append(data)

            def joint_state_cb(msg: JointState) -> None:
                _collect_positions(msg, self.joint_position_data)

            def sim_joint_state_cb(msg: JointState) -> None:
                _collect_positions(msg, self.sim_position_data)

            def robot_status_cb(msg: String) -> None:
                try:
                    data = json.loads(msg.data)
                    state = data.get("state", "IDLE")
                    note = data.get("note", "")
                    with self._data_lock:
                        self.current_mode = state
                        self.current_note = note
                except (json.JSONDecodeError, KeyError):
                    pass

            def defect_cb(msg: String) -> None:
                """Birincil konu: kusur başına tek JSON."""
                try:
                    self._upsert_defect(json.loads(msg.data))
                except (json.JSONDecodeError, TypeError, ValueError):
                    return
                self.socketio.emit("defect_list", {"defects": self._defect_snapshot()})

            def defect_list_cb(msg: String) -> None:
                """Yedek konu: tüm kusurlar tek bir JSON dizisi içinde."""
                try:
                    payloads = json.loads(msg.data)
                except (json.JSONDecodeError, TypeError, ValueError):
                    return
                if not isinstance(payloads, list):
                    return
                for payload in payloads:
                    if isinstance(payload, dict):
                        self._upsert_defect(payload)
                snapshot = self._defect_snapshot()
                self.socketio.emit("defect_list", {"defects": snapshot})
                # Bu konu algılama turu bittiğinde bir kez yayımlanır; algılamanın
                # tamamlandığının en güvenilir işareti budur.
                if snapshot:
                    self._notify_sensing_complete(len(snapshot))

            def defect_status_cb(msg: String) -> None:
                """Temizleme düğümünden gelen DETECTED/CLEANING/CLEANED güncellemelerini işler."""
                try:
                    data = json.loads(msg.data)
                    defect_id = str(data.get("defect_id", "")).strip()
                    status = str(data.get("status", "")).strip()
                except (json.JSONDecodeError, TypeError, ValueError):
                    return
                if not defect_id or not status:
                    return
                with self._data_lock:
                    if defect_id in self.defects:
                        self.defects[defect_id]["status"] = status
                self.socketio.emit("defect_list", {"defects": self._defect_snapshot()})

            def force_cb(msg: WrenchStamped) -> None:
                t = time.time()
                force_z = msg.wrench.force.z
                with self._data_lock:
                    self.force_data.append({"t": t, "force_z": force_z})

            def scenario_event_cb(msg: String) -> None:
                """
                Algılama düğümünden gelen senaryo olaylarını işler.

                Burada yalnızca CLEANING_COMPLETE işlenir. doors_mission_node tur bittiğinde
                LAST_VIEWPOINT_REACHED'i yine yayımlar -- gerçek bir olaydır ve başka araçlar
                kullanabilir -- ama bu panel onu yok sayar: KPI3 senaryo arızası kapı işine değil
                HARMONY doğrulama koşusuna aittir; bu yüzden tetiklenmeye bırakılmadı, çıkarıldı.
                """
                try:
                    data = json.loads(msg.data)
                    event = str(data.get("event", "")).strip().upper()
                except (json.JSONDecodeError, TypeError, ValueError):
                    return
                if event == "CLEANING_COMPLETE":
                    self._notify_cleaning_complete(data)

            # Kalıcı komut yayıncısı (düğme tıklamaları buradan gider).
            self._cmd_pub = node.create_publisher(String, "/harmony/cmd_input", 10)

            node.create_subscription(JointState, "/joint_states", joint_state_cb, 10)
            # Gazebo ikizinin AYNI 16 eklemin sim_ ön ekli aynası. /sim/sim_joint_states
            # DEĞİL: o konu bariyerler, masalar, kabin ve insan modeli dahil 32 ad taşır;
            # hiçbiri bir robot eklemi grafiğine ait değildir. Çalışan hücrede
            # doğrulandı.
            node.create_subscription(
                JointState, "/sim/joint_states", sim_joint_state_cb, 10)
            node.create_subscription(String, "/harmony/robot_status", robot_status_cb, 10)
            node.create_subscription(String, TOPIC_DEFECT, defect_cb, 10)
            node.create_subscription(String, TOPIC_DEFECT_LIST, defect_list_cb, latched)
            node.create_subscription(String, TOPIC_DEFECT_STATUS, defect_status_cb, 10)
            node.create_subscription(WrenchStamped, "/harmony/cleaning/force", force_cb, 10)
            node.create_subscription(String, "/harmony/scenario_event", scenario_event_cb, 10)

            node.get_logger().info("Doors dashboard listener started.")

            while self._running and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)

            self._cmd_pub = None
            node.destroy_node()
            rclpy.shutdown()

        except Exception as e:
            self._cmd_pub = None
            print(f"[ROS2DataCollector] ROS2 error: {e}")
            print("[ROS2DataCollector] ROS2 data monitoring disabled.")

    def _periodic_push(self) -> None:
        """Veriyi her 200 ms'de bir arayüze iletir."""
        while self._running:
            time.sleep(0.2)

            try:
                now = time.time()
                cutoff = now - self.buffer_seconds

                with self._data_lock:
                    pos_filtered = [d for d in self.joint_position_data if d["t"] > cutoff]
                    sim_filtered = [d for d in self.sim_position_data if d["t"] > cutoff]
                    force_filtered = [d for d in self.force_data if d["t"] > cutoff]

                    # Kip eşlemesi
                    mode_map = {
                        "SR_MODE": "Sensing Robot",
                        "CR_MODE": "Cleaning Robot",
                        "WAITING": "WAITING",
                        "IDLE": "IDLE",
                    }
                    mode_display = mode_map.get(self.current_mode, self.current_mode)
                    note = self.current_note
                    defect = len(self.defect_order)

                pos_sampled = self._downsample(pos_filtered, 100)
                sim_sampled = self._downsample(sim_filtered, 100)
                force_sampled = self._downsample(force_filtered, 100)

                self.socketio.emit("chart_data", {
                    "positions": pos_sampled,
                    "sim_positions": sim_sampled,
                    "forces": force_sampled,
                    "now": now,
                    "window": self.buffer_seconds,
                })

                self.socketio.emit("robot_info", {
                    "mode": mode_display,
                    "note": note,
                    "defect_count": defect,
                })

            except Exception as e:
                print(f"[ROS2DataCollector] Push error: {e}")

    def _downsample(
            self, data: Sequence[Dict[str, float]], max_points: int) -> Sequence[Dict[str, float]]:
        """Girdileri atlayarak basit seyreltme (downsample) yapar."""
        if len(data) <= max_points:
            return data
        step = len(data) / max_points
        return [data[int(i * step)] for i in range(max_points)]

    def stop(self) -> None:
        self._running = False


data_collector = ROS2DataCollector(socketio)


# ==============================================================================
# Komut yayıncısı (ROS 2 topic pub)
# ==============================================================================


class CommandPublisher:
    """
    Komutları /harmony/cmd_input konusuna yayımlar.

    Birincil yol, ROS2DataCollector'un kalıcı düğümü üzerinden doğrudan yayımlamaktır:
    `ros2 topic pub -1` her seferinde yeni bir düğüm kurar ve keşif tamamlanmadan
    çıkabildiği için ileti sessizce düşerdi (düğmeye birkaç kez basmak gerekiyordu).
    Alt süreç yolu yalnızca ROS başlatılamamışsa yedek olarak kullanılır.
    """

    @staticmethod
    def publish(cmd_name: str) -> None:
        if data_collector.publish_cmd(cmd_name):
            return
        print(f"[CommandPublisher] ROS unavailable, falling back to subprocess for {cmd_name}")
        CommandPublisher._publish_via_subprocess(cmd_name)

    @staticmethod
    def _publish_via_subprocess(cmd_name: str) -> None:
        cmd = (
            f'ros2 topic pub /harmony/cmd_input std_msgs/msg/String '
            f'"{{data: \'{{\\\"cmd\\\":\\\"{cmd_name}\\\"}}\'}}" -1'
        )
        full_cmd = f"{SOURCE_PREFIX} && {cmd}"

        my_env = os.environ.copy()
        if "QT_QPA_PLATFORM_PLUGIN_PATH" in my_env:
            del my_env["QT_QPA_PLATFORM_PLUGIN_PATH"]

        try:
            process = subprocess.Popen(
                full_cmd,
                shell=True,
                executable="/bin/bash",
                env=my_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Bekleme yok: gönder ve unut (zaman aşımı iş parçacığıyla)
            def _wait() -> None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            threading.Thread(target=_wait, daemon=True).start()
        except Exception as e:
            print(f"[CommandPublisher] Error publishing {cmd_name}: {e}")


cmd_publisher = CommandPublisher()


# ==============================================================================
# Kamera akışı
# ==============================================================================


class CameraStreamer:
    """
    Gazebo kamerası ve gerçek dünya kamerası için MJPEG akışı.

    Gazebo yarısında ikizin kamerası bir ROS konusu değildir; Gazebo taşıma veri yolunda
    yayımlanır, bu yüzden rclpy ile değil gz.transport ile okunur. İki yarı da sessizce
    devre dışı kalabilir: gz.transport yoksa ya da RTSP uç noktasına ulaşılamıyorsa o
    akış kapanır, diğeri çalışmaya devam eder.
    """

    def __init__(self) -> None:
        self._gazebo_frame = None
        self._real_frame = None
        self._gazebo_lock = threading.Lock()
        self._real_lock = threading.Lock()
        self._gazebo_running = False
        self._real_running = False

    def start_gazebo_stream(self) -> None:
        """Gazebo kamerasını gz.transport üzerinden başlatır."""
        self._gazebo_running = True
        thread = threading.Thread(target=self._gazebo_worker, daemon=True)
        thread.start()

    def _gazebo_worker(self) -> None:
        try:
            from gz.transport13 import Node as GzNode
            from gz.msgs10.image_pb2 import Image as GzImage
            from PIL import Image as PILImage
            import io

            def cb(msg: Any) -> None:
                try:
                    img = PILImage.frombytes("RGB", (msg.width, msg.height), msg.data)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=70)
                    with self._gazebo_lock:
                        self._gazebo_frame = buf.getvalue()
                except Exception as e:
                    print(f"[CameraStreamer] Gazebo frame decode error: {e}")

            gz_node = GzNode()
            gz_node.subscribe(GzImage, "/web_camera/image", cb)

            print("[CameraStreamer] Gazebo camera subscriber started.")
            while self._gazebo_running:
                time.sleep(0.1)

        except ImportError:
            print("[CameraStreamer] gz.transport not available. Gazebo camera disabled.")
        except Exception as e:
            print(f"[CameraStreamer] Gazebo camera error: {e}")

    def generate_gazebo(self) -> Iterator[bytes]:
        """Gazebo kamerası için MJPEG üreteci."""
        while True:
            with self._gazebo_lock:
                frame = self._gazebo_frame
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + frame + b"\r\n")
            time.sleep(0.05)

    def start_real_stream(self) -> None:
        """Gerçek dünya RTSP kamera akışını başlatır."""
        self._real_running = True
        thread = threading.Thread(target=self._real_worker, daemon=True)
        thread.start()

    def _get_rviz_window(self) -> Any:
        try:
            from Xlib import display

            def get_window_by_name(window: Any, name: str) -> Any:
                try:
                    w_name = window.get_wm_name()
                except Exception:
                    w_name = None
                if w_name and name in w_name:
                    return window
                for child in window.query_tree().children:
                    res = get_window_by_name(child, name)
                    if res:
                        return res
                return None

            d = display.Display()
            root = d.screen().root
            return get_window_by_name(root, "RViz")
        except Exception:
            return None

    def _try_rviz_capture(self, rviz_win: Any) -> Optional[bytes]:
        try:
            from Xlib import X
            from PIL import Image
            import numpy as np
            import cv2

            if rviz_win:
                geom = rviz_win.get_geometry()
                raw = rviz_win.get_image(0, 0, geom.width, geom.height, X.ZPixmap, 0xffffffff)
                img = Image.frombytes("RGB", (geom.width, geom.height), raw.data, "raw", "BGRX")
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                if geom.width > 1280:
                    scale = 1280 / geom.width
                    cv_img = cv2.resize(cv_img, (1280, int(geom.height * scale)))

                _, jpeg = cv2.imencode(".jpg", cv_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                return jpeg.tobytes()
        except Exception:
            pass
        return None

    def _real_worker(self) -> None:
        """RTSP akışını yakalayıp MJPEG'e çevirir; olmazsa RViz'e düşer."""
        try:
            import cv2
            rtsp_url = "rtsp://192.168.3.51:554/live/0"

            while self._real_running:
                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    print(f"[CameraStreamer] RTSP connection failed: {rtsp_url}")
                    print("[CameraStreamer] Fallback: Searching for RViz window...")

                    start_time = time.time()
                    rviz_win = self._get_rviz_window()

                    while self._real_running and (time.time() - start_time) < 5:
                        if not rviz_win:
                            rviz_win = self._get_rviz_window()

                        rviz_frame = self._try_rviz_capture(rviz_win) if rviz_win else None

                        if rviz_frame:
                            with self._real_lock:
                                self._real_frame = rviz_frame
                            time.sleep(0.05)
                        else:
                            time.sleep(1.0)
                    continue

                print("[CameraStreamer] Real camera connected.")
                while self._real_running:
                    ret, frame = cap.read()
                    if not ret:
                        print("[CameraStreamer] Failed to capture RTSP frame, reconnecting...")
                        break
                    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    with self._real_lock:
                        self._real_frame = jpeg.tobytes()

                cap.release()
                time.sleep(2)

        except ImportError:
            print("[CameraStreamer] OpenCV not available. Real camera disabled.")
        except Exception as e:
            print(f"[CameraStreamer] Real camera error: {e}")

    def generate_real(self) -> Iterator[bytes]:
        """Gerçek kamera için MJPEG üreteci."""
        while True:
            with self._real_lock:
                frame = self._real_frame
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + frame + b"\r\n")
            time.sleep(0.05)

    def stop(self) -> None:
        self._gazebo_running = False
        self._real_running = False


camera_streamer = CameraStreamer()


# ==============================================================================
# Flask yolları (route)
# ==============================================================================


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/stream/gazebo")
def gazebo_stream() -> Response:
    return Response(
        camera_streamer.generate_gazebo(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stream/real")
def real_stream() -> Response:
    return Response(
        camera_streamer.generate_real(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status() -> Response:
    return jsonify(process_mgr.get_status())


# ==============================================================================
# SocketIO olayları
# ==============================================================================


@socketio.on("connect")
def handle_connect() -> None:
    emit("status_update", process_mgr.get_status())
    # Geç bağlanan istemci mevcut kusur tablosunu hemen görsün.
    emit("defect_list", {"defects": data_collector._defect_snapshot()})


@socketio.on("start_hil")
def handle_start_hil(data: Dict[str, Any]) -> None:
    use_fake_hardware = data.get("use_fake_hardware", False)
    process_mgr.start_hil(use_fake_hardware)


@socketio.on("confirm_robot")
def handle_confirm_robot(data: Optional[Dict[str, Any]] = None) -> None:
    data = data or {}
    process_mgr.confirm_robot_ready(
        only_sim=bool(data.get("only_sim", True)),
        force_replan=bool(data.get("force_replan", False)),
    )


@socketio.on("stop_all")
def handle_stop_all() -> None:
    thread = threading.Thread(target=process_mgr.stop_all, daemon=True)
    thread.start()


@socketio.on("reset_mission")
def handle_reset_mission() -> None:
    """Arayüzü sıfırlar (kusurlar, sayaçlar, kip, grafik/günlük gösterimi)."""
    data_collector.reset_ui()
    process_mgr._emit_log("SYSTEM", "🔄 Arayüz sıfırlandı — yeni mission'a hazır.")


#: /harmony/cmd_input üzerinden yayımlanabilen komutlar
VALID_CMDS = ("START", "CONFIRM", "REINSPECT", "STOP")


@socketio.on("publish_cmd")
def handle_publish_cmd(data: Dict[str, Any]) -> None:
    cmd = data.get("cmd", "")
    if cmd in VALID_CMDS:
        # Yeni bir tarama başlıyorsa eski kusur tablosunu temizle.
        if cmd == "START":
            data_collector.clear_defects()
        cmd_publisher.publish(cmd)
        process_mgr._emit_log("SYSTEM", f"📡 Published command: {cmd}")
    else:
        process_mgr._emit_log("SYSTEM", f"❌ Unknown command: {cmd}")


# ==============================================================================
# Ana giriş noktası
# ==============================================================================


def main() -> None:
    print("=" * 60)
    print("  Doors — Multirobot Sensing / Cleaning Dashboard")
    print("  http://localhost:8082")
    print("=" * 60)

    # Kamera akışlarını başlat
    camera_streamer.start_gazebo_stream()
    camera_streamer.start_real_stream()

    # ROS 2 veri toplayıcısını başlat
    data_collector.start()

    try:
        # 8081 değil 8082: HARMONY paneli o portu kullanıyor; böylece ikisi birlikte çalışabilir.
        socketio.run(app, host="0.0.0.0", port=8082, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        camera_streamer.stop()
        data_collector.stop()
        process_mgr.stop_all()


if __name__ == "__main__":
    main()
