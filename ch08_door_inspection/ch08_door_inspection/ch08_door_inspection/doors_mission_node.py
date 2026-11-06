#!/usr/bin/env python3
"""
Kapı görevi yöneticisi; book_pymoveit2 örneklerindeki sensing_robot.py'nin kapı karşılığı.

NEDEN VAR
─────────
HARMONY operatör paneli her şeyi /harmony/cmd_input üzerinden sürer ve
/harmony/robot_status, kusur konuları ve /harmony/scenario_event'i dinler. Algılama
tarafı START bekleyen UZUN ÖMÜRLÜ bir düğümdür.

Kapı muayenesi bunun tersidir: ur_inspection_node ve kawasaki_inspection_node ayağa
kalktıkları anda hareket etmeye başlar (inspection_base.run()'ın komut kapısı yoktur)
ve tur bitince süreçler ÇIKAR. Hiçbir durum bilgisi yayımlamazlar; yalnızca günlük
satırları yazarlar.

Bu düğüm ikisini birbirine bağlar. sensing_robot.py ile tamamen aynı konu
sözleşmesini konuşur; böylece panel değişmeden çalışır; ama bir kolu kendisi sürmek
yerine ch08_door_inspection başlatmasını bir alt süreç olarak yönetir:

    START       -> `ros2 launch ch08_door_inspection doors_inspection.launch.py` başlat
                   ve stdout'unu /harmony/robot_status ilerlemesine çevir
    tur biter   -> LAST_VIEWPOINT_REACHED (panelin KPI3 hata penceresi), ardından
                   dört senaryo kusurunu + RViz işaretçilerini yayımla
    CONFIRM     -> cleaning_mission_runner'a devret (yalnızca UR'yi sürer)
    STOP        -> çalışan yörüngeleri iptal et, KOLLARIN DURMASINI BEKLE ve ancak
                   ondan sonra başlatmayı kapat

ÖLÇÜLDÜ; BİTİŞ ALGILAMASININ BU HÂLDE OLMASININ SEBEBİ: muayene başlatması tur
bittiğinde ÇIKMAZ. multirobot_inspection.launch.py kalıcı bir düğüm olan
multirobot_viewpoint_visualizer'ı da başlatır ve hiçbir on_exit işleyicisi kaydetmez;
bu yüzden iki kol yürütücüsü de çıktıktan sonra `ros2 launch` süresiz açık kalır
(süreç listesiyle doğrulandı: iki *_inspection_node süreci de gitti, launch ve
görselleştirici hâlâ yaşıyor). Başlatma sürecini beklemek bu yüzden sonsuza kadar
asılı kalırdı. Bunun yerine tur, kol yürütücü süreçleri belirip sonra kaybolduğunda
bitmiş sayılır ve bu düğüm başlatmayı kendisi kapatır.

Burada bilerek YAPILMAYAN: muayene düğümlerini bir komuta bağlamak. Bu, doğrulanmış
şase işinin paylaştığı ch07_multirobot_planner'ı düzenlemek demek olurdu. Bu
dosyadaki her şey eklemedir.

Kusur koordinatları book_pymoveit2.harmony_defects'ten gelir; HARMONY senaryosunun
kullandığı AYNI modül; iki senaryo asla birbirinden kaymasın diye kopyalanmak yerine
içe aktarılır. Bu dört kusur zaten kapıların üzerindedir (dördünün x değerleri de
kapının sınır kutusunun içine düşer); temizleme tarafının değişiklik gerektirmemesinin
sebebi budur.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)

from action_msgs.srv import CancelGoal
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

# Senaryo kusurlarının tek doğru kaynağı; HARMONY ile ortaktır.
from book_pymoveit2 import harmony_defects as hd


def _now_ros(node: Node) -> str:
    t = node.get_clock().now().to_msg()
    return f"{t.sec}.{t.nanosec:09d}"


def _latched_qos(depth: int = 1) -> QoSProfile:
    """Sonradan bağlanan bir panel ya da RViz son mesajı yine görsün diye."""
    return QoSProfile(
        depth=depth,
        history=QoSHistoryPolicy.KEEP_LAST,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


#: "[UR 4/9] Moving UR->ur_vp_004."  — inspection_base.run_sequence()
RE_PROGRESS = re.compile(r"\[(\w[\w ]*?) (\d+)/(\d+)\]\s+Moving")
#: "[UR] sequence complete: 9/9 viewpoints captured."  — inspection_base.run_sequence()
RE_COMPLETE = re.compile(r"\[(\w[\w ]*?)\]\s+sequence complete:\s*(\d+)/(\d+)")


class DoorsMissionNode(Node):

    def __init__(self) -> None:
        super().__init__("doors_mission_node")

        # ---------------- Parametreler ----------------
        self.declare_parameter("fixed_frame", "world")
        # doors_inspection.launch.py'ye iletilir. only_sim HANGİ KAMERALARIN verisinin
        # toplanacağını belirler; HIL başlatmasının use_fake_hardware'iyle (ROBOTUN gerçek
        # olup olmadığına karar verir) aynı şey değildir, ama sahte donanımla tek mantıklı
        # birleşim only_sim=true'dur.
        self.declare_parameter("only_sim", True)
        # Önbellekli yörüngeler çarpışma yeniden denetlenmeden oynatılır; bu yüzden
        # çarpışma modelindeki her değişiklikten sonra bu bir kez açılmalıdır.
        self.declare_parameter("force_replan", False)
        # Kaçış yolu: eklenecek başka her şey, örneğin "kawasaki_velocity:=0.05".
        self.declare_parameter("extra_inspection_args", "")
        self.declare_parameter("defect_publish_interval", 0.3)
        self.declare_parameter("marker_publish_rate_hz", 1.0)
        # Durum kalp atışı. Olmazsa robot_status yalnızca durum değişikliklerinde
        # yayımlanır; koşunun ortasında bağlanan (ya da yeniden bağlanan) bir panel bir
        # sonraki olaya kadar hiçbir şey görmez. sensing_robot.py bunu TCP pozu
        # zamanlayıcısından yeniden yayımlayarak önler; bu aynı fikrin açık hâlidir.
        self.declare_parameter("status_heartbeat_hz", 1.0)
        # STOP yolu. Bunların varsayılanları muayene düğümlerinin kullandıklarıyla aynıdır;
        # böylece buradaki bir iptal onların gönderdiği hedeflere ulaşır.
        self.declare_parameter("ur_controller_action",
                               "/scaled_joint_trajectory_controller/follow_joint_trajectory")
        self.declare_parameter("kawasaki_controller_action",
                               "/kawasaki/kawasaki_controller/follow_joint_trajectory")
        self.declare_parameter("stop_settle_timeout_s", 8.0)
        self.declare_parameter("stop_still_velocity", 0.01)   # rad/s ya da m/s
        # Başlatma başladıktan sonra kol yürütücülerinin belirmesine tanınan süre.
        # Cömert: önce move_group keşfi ve planlama sahnesi kurulumu yapılır.
        self.declare_parameter("inspection_start_timeout_s", 90.0)

        self.fixed_frame = self.get_parameter("fixed_frame").value
        self.defect_publish_interval = float(
            self.get_parameter("defect_publish_interval").value)
        marker_rate = float(self.get_parameter("marker_publish_rate_hz").value)

        # ---------------- Yayıncılar ----------------
        self.robot_status_pub = self.create_publisher(String, "/harmony/robot_status", 10)
        self.scenario_event_pub = self.create_publisher(String, "/harmony/scenario_event", 10)
        self.defect_pub = self.create_publisher(String, hd.TOPIC_DEFECT, 10)
        self.defect_list_pub = self.create_publisher(
            String, hd.TOPIC_DEFECT_LIST, _latched_qos())
        self.marker_pub = self.create_publisher(
            MarkerArray, hd.TOPIC_DEFECT_MARKERS, _latched_qos())

        # ---------------- Aboneler ----------------
        cbg = ReentrantCallbackGroup()
        self.create_subscription(String, "/harmony/cmd_input", self._cmd_cb, 10,
                                 callback_group=cbg)
        self.create_subscription(String, hd.TOPIC_DEFECT_STATUS,
                                 self._defect_status_cb, 10, callback_group=cbg)
        # Yalnızca bir STOP'tan sonra kolların gerçekten durduğunu doğrulamak için.
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10,
                                 callback_group=cbg)

        # Hepsini iptal eden istemciler. Eylem tanımı, sıfır goal_id ve sıfır zaman damgalı
        # bir CancelGoal isteğini "bu sunucudaki her hedefi iptal et" olarak ele alır; bu
        # yüzden düz bir servis istemcisi yeter; ActionClient ve hedef tutamacı kaydı gerekmez.
        self._cancel_clients = {}
        for tag, param in (("UR", "ur_controller_action"),
                           ("Kawasaki", "kawasaki_controller_action")):
            action = self.get_parameter(param).value
            self._cancel_clients[tag] = self.create_client(
                CancelGoal, f"{action}/_action/cancel_goal", callback_group=cbg)

        # ---------------- Durum ----------------
        self._mode_lock = threading.Lock()
        self.state = "IDLE"
        self.mode = "IDLE"
        self._note = "Waiting for START"
        self._level = "INFO"

        self._proc: Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()
        self.start_requested = threading.Event()
        self.stop_requested = threading.Event()

        self._marker_lock = threading.Lock()
        self._defects_reported = False
        self._defect_status = {}

        # kol etiketi -> (toplanan, planlanan); başlatmanın günlük satırlarından ayıklanır.
        self._result_lock = threading.Lock()
        self._arm_results = {}

        self._vel_lock = threading.Lock()
        self._max_abs_vel = 0.0
        self._vel_stamp = 0.0

        if marker_rate > 0:
            self.create_timer(1.0 / marker_rate, self._publish_markers)
        hb = float(self.get_parameter("status_heartbeat_hz").value)
        if hb > 0:
            self.create_timer(1.0 / hb, self._heartbeat)

        self.worker = threading.Thread(target=self._run_state_machine, daemon=True)
        self.worker.start()

        self._publish_robot_status("IDLE", "IDLE", "Waiting for START")
        self.get_logger().info(
            f"DoorsMissionNode ready | only_sim={self.get_parameter('only_sim').value} "
            f"| force_replan={self.get_parameter('force_replan').value}")

    # ================================================================== #
    # Durum
    # ================================================================== #
    def _publish_robot_status(self, state: str, mode: str, note: str, level: str = "INFO") -> None:
        # Kalp atışı son satırı aynen tekrarlayabilsin diye sakla.
        with self._mode_lock:
            self._note = note
            self._level = level
        payload = {
            "timestamp": _now_ros(self),
            "state": state,
            "mode": mode,
            "level": level,
            "note": note,
            "frame_id": self.fixed_frame,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.robot_status_pub.publish(msg)

    def _heartbeat(self) -> None:
        with self._mode_lock:
            state, mode, note, level = self.state, self.mode, self._note, self._level
        self._publish_robot_status(state, mode, note, level=level)

    def _set_state(self, state: str, mode: str, note: str, level: str = "INFO") -> None:
        with self._mode_lock:
            self.state = state
            self.mode = mode
        self._publish_robot_status(state, mode, note, level=level)
        self.get_logger().info(f"State change: {state}/{mode} - {note}")

    def _publish_scenario_event(
            self, event: str, note: str = "", extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {"event": event, "note": note, "timestamp": _now_ros(self)}
        if extra:
            payload.update(extra)
        msg = String()
        msg.data = json.dumps(payload)
        self.scenario_event_pub.publish(msg)
        self.get_logger().info(f"Scenario event published: {event}")

    # ================================================================== #
    # Komutlar
    # ================================================================== #
    def _cmd_cb(self, msg: String) -> None:
        try:
            cmd = str(json.loads(msg.data).get("cmd", "")).upper().strip()
        except Exception:
            self.get_logger().warning(
                f"Invalid JSON on /harmony/cmd_input: {msg.data[:200]}")
            return

        self.get_logger().info(f"Command received: {cmd}")

        if cmd in ("START", "REINSPECT"):
            if self._inspection_running():
                self.get_logger().warning(
                    f"{cmd} ignored: an inspection is already running.")
                return
            self.stop_requested.clear()
            self._reset_defects()
            self.start_requested.set()
            self._set_state("SR_MODE", "SR", f"{cmd} received, doors inspection begins")

        elif cmd == "STOP":
            self.stop_requested.set()
            self.start_requested.clear()
            # Durdurma dizisini geri çağırma iş parçacığı dışında çalıştır: iptal etmek ve
            # kolları beklemek saniyeler sürer ve yürütücüyü engellememelidir.
            threading.Thread(target=self._stop_sequence, daemon=True).start()

        elif cmd == "CONFIRM":
            self._set_state("CR_MODE", "CR",
                            "CONFIRM received (cleaning handled by CR node)")

        elif cmd == "WAITING":
            self._set_state("WAITING", "SR", "WAITING command received, staying in WAITING")

        else:
            self.get_logger().warning(f"Unknown command: {cmd}")

    # ================================================================== #
    # Kusurlar / işaretçiler (RViz aynı davransın diye sensing_robot.py'nin aynası)
    # ================================================================== #
    def _reset_defects(self) -> None:
        with self._marker_lock:
            self._defects_reported = False
            self._defect_status = {}

    def _defect_status_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            defect_id = str(data.get("defect_id", "")).strip()
            status = str(data.get("status", "")).strip()
        except Exception:
            return
        if not defect_id or not status:
            return
        with self._marker_lock:
            self._defect_status[defect_id] = status

    def _publish_markers(self) -> None:
        """
        İşaretçi konusunu her zaman canlı tutar.

        Algılama raporlamadan önce DELETEALL, sonrasında gerçek dizi yayımlanır. Böylece
        RViz konuyu baştan seçebilir ve kusurlar senaryo söylemeden görünmez.
        """
        stamp = self.get_clock().now().to_msg()
        with self._marker_lock:
            reported = self._defects_reported
            statuses = dict(self._defect_status)
        if not reported:
            self.marker_pub.publish(
                hd.build_empty_marker_array(frame_id=self.fixed_frame, stamp=stamp))
        else:
            self.marker_pub.publish(
                hd.build_marker_array(statuses=statuses, frame_id=self.fixed_frame,
                                      stamp=stamp))

    def _publish_defects(self) -> None:
        stamp_iso = datetime.now().isoformat()
        payloads = hd.all_payloads(frame_id=self.fixed_frame, timestamp=stamp_iso)

        for payload in payloads:
            msg = String()
            msg.data = json.dumps(payload)
            self.defect_pub.publish(msg)
            p = payload["position"]
            self.get_logger().info(
                f"Defect published: {payload['defect_id']} ({payload['defect_type']}) "
                f"x={p['x']:.3f} y={p['y']:.3f} z={p['z']:.3f}")
            if self.defect_publish_interval > 0:
                time.sleep(self.defect_publish_interval)

        list_msg = String()
        list_msg.data = json.dumps(payloads)
        self.defect_list_pub.publish(list_msg)

        with self._marker_lock:
            self._defects_reported = True
            self._defect_status = {p["defect_id"]: p["status"] for p in payloads}
        self._publish_markers()

        self._publish_robot_status(
            "SR_MODE", "SR", f"Sensing complete. {len(payloads)} defects reported")

    # ================================================================== #
    # Muayene alt süreci
    # ================================================================== #
    def _inspection_running(self) -> bool:
        with self._proc_lock:
            return self._proc is not None and self._proc.poll() is None

    def _build_inspection_cmd(self) -> List[str]:
        only_sim = "true" if self.get_parameter("only_sim").value else "false"
        force_replan = "true" if self.get_parameter("force_replan").value else "false"
        cmd = ["ros2", "launch", "ch08_door_inspection", "doors_inspection.launch.py",
               f"only_sim:={only_sim}", f"force_replan:={force_replan}"]
        extra = str(self.get_parameter("extra_inspection_args").value or "").strip()
        if extra:
            cmd += shlex.split(extra)
        return cmd

    def _spawn_inspection(self) -> bool:
        cmd = self._build_inspection_cmd()
        self.get_logger().info("Launching inspection: " + " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                # Kendi süreç grubu; böylece başlatma ağacının tamamına tek seferde sinyal
                # gönderilebilir.
                preexec_fn=os.setsid,
            )
        except Exception as e:
            self.get_logger().error(f"Failed to launch ch08_door_inspection: {e}")
            self._set_state("IDLE", "IDLE", f"Inspection launch failed: {e}", level="ERROR")
            return False
        with self._proc_lock:
            self._proc = proc
        return True

    def _consume_inspection_output(self, proc: subprocess.Popen) -> None:
        """
        Muayene günlüğünü robot_status ilerlemesine çevirir.

        Her kolun son sayımı da kaydedilir.

        stdout'u okumak, bu düğümü ch07_multirobot_planner'ın dışında tutan şeydir: burada
        eşleşen iki günlük satırı zaten vardı; yani bir ilerleme göstergesi için ortak kodda
        değişiklik gerekmedi. Bu satırların ifadesi bir gün değişirse ilerleme sessizce
        "running"e düşer ve tamamlanma sayımı boş kalır; koşunun kendisi ve bitiş algılaması
        (metni değil süreçleri izler) etkilenmez.

        Kendi iş parçacığında çalışır: turun bittiği, muayene düğümü süreçlerini yoklayarak
        algılanır ve bu, stdout hâlâ boşaltılırken yapılmalıdır.
        """
        if proc.stdout is None:
            return
        per_arm = {}
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                line = line.rstrip()
                if self.stop_requested.is_set():
                    # Çocuk süreç dolu bir boruda takılmasın diye boşaltmaya devam et, ama
                    # kapatılmakta olan bir koşunun ilerlemesini raporlamayı bırak.
                    continue

                m = RE_PROGRESS.search(line)
                if m:
                    arm, idx, total = m.group(1), int(m.group(2)), int(m.group(3))
                    per_arm[arm] = (idx, total)
                    note = "  ".join(f"{a} {i}/{n}" for a, (i, n) in sorted(per_arm.items()))
                    self._publish_robot_status("SR_MODE", "SR", f"Scanning: {note}")
                    continue

                m = RE_COMPLETE.search(line)
                if m:
                    arm, saved, total = m.group(1), int(m.group(2)), int(m.group(3))
                    with self._result_lock:
                        self._arm_results[arm] = (saved, total)
                    self._publish_robot_status(
                        "SR_MODE", "SR", f"{arm} finished: {saved}/{total} viewpoints")
        except (ValueError, OSError):
            pass

    def _inspection_node_pids(self, pgid: int) -> List[str]:
        """Başlatmanın süreç grubunda hâlâ yaşayan kol yürütücülerinin PID'leri."""
        try:
            out = subprocess.run(
                ["pgrep", "-g", str(pgid), "-f", "inspection_node"],
                capture_output=True, text=True, timeout=5).stdout
        except (subprocess.SubprocessError, OSError):
            return []
        return [p for p in out.split() if p]

    def _await_tour_end(self, proc: subprocess.Popen) -> str:
        """
        İki kol yürütücüsü bitene kadar bekler.

        NEDEN BAŞLATMANIN ÇIKMASI BEKLENMİYOR: multirobot_inspection.launch.py kalıcı bir
        düğüm olan multirobot_viewpoint_visualizer'ı da başlatır ve başlatma hiçbir on_exit
        işleyicisi kaydetmez. Yani `ros2 launch` iki kol bittikten çok sonra da çalışmaya
        devam eder; onu beklemek sonsuza kadar asılı kalırdı. Yürütücü süreçlerini izlemek
        bundan da günlük ifadelerinden de bağımsızdır.

        "done", "stopped" ya da "died" döndürür.
        """
        pgid = os.getpgid(proc.pid)
        grace_deadline = time.monotonic() + float(
            self.get_parameter("inspection_start_timeout_s").value)
        seen_nodes = False

        while True:
            if self.stop_requested.is_set():
                return "stopped"
            if proc.poll() is not None:
                return "died"

            pids = self._inspection_node_pids(pgid)
            if pids:
                seen_nodes = True
            elif seen_nodes:
                # Ayağa kalktılar ve şimdi hepsi gitti: tur öyle ya da böyle bitti. stdout'un
                # son satırları boşaltması için kısa bir süre tanı.
                time.sleep(1.0)
                return "done"
            elif time.monotonic() > grace_deadline:
                self.get_logger().error(
                    "Inspection nodes never appeared; is the launch failing?")
                return "died"

            time.sleep(1.0)

    def _run_state_machine(self) -> None:
        while rclpy.ok():
            if not self.start_requested.wait(timeout=0.5):
                continue
            self.start_requested.clear()
            if self.stop_requested.is_set():
                continue

            self._publish_robot_status("SR_MODE", "SR", "Starting doors inspection")
            with self._result_lock:
                self._arm_results = {}
            if not self._spawn_inspection():
                continue

            with self._proc_lock:
                proc = self._proc
            reader = threading.Thread(
                target=self._consume_inspection_output, args=(proc,), daemon=True)
            reader.start()

            outcome = self._await_tour_end(proc)

            # Başlatma kendiliğinden asla çıkmaz (görselleştirici kalıcıdır); bu yüzden
            # yönetici onu her zaman kendisi kapatır. Açık bırakmak bir sonraki START'ın aynı
            # konuda ikinci bir görselleştiriciyle çakışmasına da yol açardı.
            self._teardown_inspection(cancel_first=(outcome == "stopped"))
            reader.join(timeout=3.0)

            with self._result_lock:
                results = dict(self._arm_results)
            captured = sum(s for s, _ in results.values())
            summary = ", ".join(f"{a} {s}/{t}" for a, (s, t) in sorted(results.items()))

            if outcome == "stopped" or self.stop_requested.is_set():
                self._set_state("IDLE", "IDLE", "Inspection stopped by operator",
                                level="WARN")
                continue

            if outcome == "died":
                self._set_state("IDLE", "IDLE",
                                "Inspection launch ended unexpectedly", level="ERROR")
                continue

            if not results or captured == 0:
                # Kollar çalıştı ama hiçbir şey toplamadı: eksik plan dosyası, erişilemeyen
                # denetleyici, /joint_states yok. Başarısızlık sayılır; böylece operatöre hiç
                # gerçekleşmemiş bir turun kusurları asla gösterilmez.
                self._set_state(
                    "IDLE", "IDLE",
                    "Inspection captured no viewpoints; check the launch log",
                    level="ERROR")
                continue

            self._publish_robot_status("SR_MODE", "SR",
                                       f"Inspection complete: {summary}")

            # Turun tamamının bitmesi, HARMONY'nin "son bakış noktasına varıldı" olayının kapı
            # karşılığıdır: iki kolla tek bir son bakış noktası yoktur; bu yüzden turun sonu tek
            # belirsizlik içermeyen andır.
            self._publish_scenario_event(
                "LAST_VIEWPOINT_REACHED",
                "Doors inspection tamamlandı, defect'ler yayınlanıyor")
            self._publish_defects()
            self._set_state("WAITING", "SR", "Waiting for CONFIRM (or REINSPECT/STOP)")

    # ================================================================== #
    # STOP (durdurma)
    # ================================================================== #
    def _joint_state_cb(self, msg: JointState) -> None:
        if not msg.velocity:
            return
        with self._vel_lock:
            self._max_abs_vel = max(abs(v) for v in msg.velocity)
            self._vel_stamp = time.monotonic()

    def _cancel_all_trajectories(self) -> None:
        """
        İki kol denetleyicisindeki her hedefi iptal eder.

        Boş bir CancelGoal isteği (sıfır goal_id, sıfır zaman damgası) eylem tanımına göre
        "hepsini iptal et" demektir; böylece bu düğüm muayene düğümlerinin gönderdiği
        hedeflere onlar hakkında hiçbir şey bilmeden ulaşır.
        """
        for tag, client in self._cancel_clients.items():
            if not client.service_is_ready():
                if not client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().warning(
                        f"{tag}: cancel service unavailable; skipping cancel.")
                    continue
            try:
                client.call_async(CancelGoal.Request())
                self.get_logger().info(f"{tag}: cancel-all sent.")
            except Exception as e:
                self.get_logger().warning(f"{tag}: cancel failed: {e}")

    def _wait_until_still(self) -> bool:
        """Bildirilen eklem hızları durulana ya da zaman aşımı dolana kadar bekler."""
        timeout = float(self.get_parameter("stop_settle_timeout_s").value)
        thresh = float(self.get_parameter("stop_still_velocity").value)
        deadline = time.monotonic() + timeout
        still_since = None
        while time.monotonic() < deadline:
            with self._vel_lock:
                vmax, stamp = self._max_abs_vel, self._vel_stamp
            fresh = stamp > 0 and (time.monotonic() - stamp) < 1.0
            if fresh and vmax < thresh:
                still_since = still_since or time.monotonic()
                if time.monotonic() - still_since > 0.5:
                    self.get_logger().info(
                        f"Arms settled (max |v| = {vmax:.4f}).")
                    return True
            else:
                still_since = None
            time.sleep(0.05)
        self.get_logger().warning(
            "Arms did not report a settled velocity before the timeout; "
            "tearing the launch down anyway.")
        return False

    def _teardown_inspection(self, cancel_first: bool) -> None:
        """
        Muayene başlatmasını kapatır ve unutur.

        cancel_first STOP yolu içindir: gerçek donanımda yalnızca başlatmayı öldürmek,
        denetleyicinin ZATEN yürüttüğü hedefe dokunmaz; kol, onu iptal edecek hiçbir şey
        kalmadan hareket etmeye devam ederdi. Normal yolda kollar bitirmiştir; iptal edilecek
        bir şey yoktur ve fazladan servis çağrısı yalnızca temizlemeye devri yavaşlatırdı.
        """
        with self._proc_lock:
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        if cancel_first:
            self._cancel_all_trajectories()
            self._wait_until_still()
        if proc.poll() is None:
            self._kill_process(proc)

    def _stop_sequence(self) -> None:
        """Yörüngeleri iptal eder, kolların durduğunu doğrular ve sonra başlatmayı kapatır."""
        self._publish_robot_status("SR_MODE", "SR",
                                   "STOP received, cancelling trajectories", level="WARN")
        self._teardown_inspection(cancel_first=True)
        self._set_state("IDLE", "IDLE", "STOP complete, back to IDLE", level="WARN")

    def _kill_process(self, proc: subprocess.Popen) -> None:
        for sig, name, wait_s in ((signal.SIGINT, "SIGINT", 10),
                                  (signal.SIGTERM, "SIGTERM", 5),
                                  (signal.SIGKILL, "SIGKILL", 3)):
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except (ProcessLookupError, OSError):
                return
            self.get_logger().info(f"Inspection launch: sent {name}.")
            try:
                proc.wait(timeout=wait_s)
                return
            except subprocess.TimeoutExpired:
                continue

    # ================================================================== #
    def destroy_node(self) -> None:
        with self._proc_lock:
            running = self._proc is not None and self._proc.poll() is None
        if running:
            self.get_logger().info("Shutting down: stopping the inspection launch.")
            self._teardown_inspection(cancel_first=True)
        super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = DoorsMissionNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
