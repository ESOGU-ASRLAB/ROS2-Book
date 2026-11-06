#!/usr/bin/env python3
"""
Temizleme (Cleaning) düğümü - v5 (yalnızca gerçek robot, HARMONY doğrulaması).

- Algılama düğümünün yayımladığı 4 yapay kusuru toplar.
- CONFIRM gelince kusurlara y'ye göre artan sırayla gider (DEF-03 -> DEF-02 ->
  DEF-01 -> DEF-04); aralarda eve (HOME) dönmez, en sonda eve döner.
- Her kusur için harmony_defects.DEFECT_JOINTS içindeki sabit eklem açıları
  kullanılır (sahada tekrarlanabilirlik için; çalışma anında IK yok).
- Her kusurda durum yayımlanır (CLEANING -> CLEANED) ve RViz işaretçileri buna
  göre renk değiştirir.
- Temizleme sırasında gelen CONFIRM yok sayılır (iki tur sorununu çözer).
- Yörünge doğrudan gerçek robotta (book_pymoveit2) planlanır ve çalıştırılır.

Bölüm 8'de kapı görevinin temizleme tarafıdır (bkz. doors_mission.launch.py).
"""

from __future__ import annotations

from threading import Thread, Event, Lock
import time
import json
import random
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from std_msgs.msg import String
from geometry_msgs.msg import WrenchStamped
from trajectory_msgs.msg import JointTrajectory

from book_pymoveit2 import MoveIt2 as MoveIt2_Real
from book_pymoveit2.robots import ur as realrobot
from book_pymoveit2 import harmony_defects as hd
from book_pymoveit2.trajectory_store import TrajectoryStore

from ch08_door_inspection import paths


def _now_ros(node: Node) -> str:
    t = node.get_clock().now().to_msg()
    return f"{t.sec}.{t.nanosec:09d}"


def _latched_qos(depth: int = 1) -> QoSProfile:
    """Geç bağlanan abonelerin (arayüz, RViz) son mesajı görmesi için."""
    return QoSProfile(
        depth=depth,
        history=QoSHistoryPolicy.KEEP_LAST,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


# ===========================================================================
# Temizleme düğümü
# ===========================================================================
class HarmonyCleaningMissionRunnerV5(Node):
    def __init__(self) -> None:
        super().__init__("harmony_cleaning_runner")

        # ---------------- Parametreler ----------------
        self.declare_parameter("fixed_frame", "world")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("approach_height_offset", 0.3)
        self.declare_parameter("cleaning_duration", 1.5)
        self.declare_parameter("paint_y_offset", 0.30)
        self.declare_parameter("paint_x_span", 0.2)
        self.declare_parameter("move_max_retries", 5)
        self.declare_parameter("move_retry_sleep_s", 0.2)
        self.declare_parameter("cartesian_max_step", 0.01)
        self.declare_parameter("cartesian_fraction_threshold", 0.40)
        self.declare_parameter("force_rate_hz", 30.0)
        self.declare_parameter("skip_if_no_defects", True)
        self.declare_parameter("home_joints", list(hd.HOME_JOINTS))
        # Bir kusura ulaşılamazsa devam et (True) ya da görevi kes (False).
        self.declare_parameter("continue_on_move_failure", True)

        # Gerçek robot hareket parametreleri
        self.declare_parameter("planner_id", "ESTkConfigDefault")
        self.declare_parameter("real_robot_velocity", 0.1)
        self.declare_parameter("real_robot_acceleration", 0.1)

        # Yörünge kayıt/oynatma: rastgeleliği önlemek için yörüngeler bir
        # kez hesaplanıp JSON'a kaydedilir, sonra hep aynıları oynatılır.
        # Varsayılan klasör ~/ros2_book_output/ch08/trajectories'tir: ilk koşu
        # yörüngeleri planlayıp cleaning.json'a kaydeder, sonrakiler oynatır.
        self.declare_parameter(
            "trajectory_dir",
            paths.output_path("trajectories"),
        )
        self.declare_parameter("trajectory_enable", True)

        self.fixed_frame = str(self.get_parameter("fixed_frame").value)
        self.tool_frame = str(self.get_parameter("tool_frame").value)
        self.approach_offset = float(self.get_parameter("approach_height_offset").value)
        self.cleaning_duration = float(self.get_parameter("cleaning_duration").value)
        self.paint_y_offset = float(self.get_parameter("paint_y_offset").value)
        self.paint_x_span = float(self.get_parameter("paint_x_span").value)
        self.move_max_retries = int(self.get_parameter("move_max_retries").value)
        self.move_retry_sleep_s = float(self.get_parameter("move_retry_sleep_s").value)
        self.cartesian_max_step = float(self.get_parameter("cartesian_max_step").value)
        self.cartesian_fraction_threshold = float(
            self.get_parameter("cartesian_fraction_threshold").value)
        self.force_rate_hz = float(self.get_parameter("force_rate_hz").value)
        self.skip_if_no_defects = bool(self.get_parameter("skip_if_no_defects").value)
        self.home_joints = [float(x) for x in self.get_parameter("home_joints").value]
        self.continue_on_move_failure = bool(self.get_parameter("continue_on_move_failure").value)

        planner_id = str(self.get_parameter("planner_id").value)
        real_velocity = float(self.get_parameter("real_robot_velocity").value)
        real_accel = float(self.get_parameter("real_robot_acceleration").value)

        trajectory_dir = str(self.get_parameter("trajectory_dir").value)
        trajectory_enable = bool(self.get_parameter("trajectory_enable").value)

        # ---------------- Yörünge deposu ----------------
        # 'cleaning' moduna özel tek JSON dosyası. Dosya varsa planlama atlanır.
        self.traj_store: Optional[TrajectoryStore] = None
        if trajectory_enable:
            self.traj_store = TrajectoryStore(
                trajectory_dir, "cleaning", logger=self.get_logger()
            )

        cbg = ReentrantCallbackGroup()

        # ---------------- MoveIt2 (gerçek) ----------------
        # MoveIt2'ye AYRI bir düğüm veriyoruz. Aynı düğüm paylaşılırsa
        # /harmony/cmd_input (STOP dahil) ve kusur abonelikleri ilk plan()
        # çağrısından sonra sessizce mesaj almaz.
        self.moveit_node = Node("harmony_cleaning_runner_moveit")
        self.moveit2 = MoveIt2_Real(
            node=self.moveit_node,
            joint_names=realrobot.joint_names(),
            base_link_name="world",
            end_effector_name=realrobot.end_effector_name(),
            group_name=realrobot.MOVE_GROUP_ARM,
            callback_group=cbg,
        )
        self.moveit2.planner_id = planner_id
        # Algılama robotuyla (sensing_robot.py) aynı hız/ivme ölçeği.
        self.moveit2.max_velocity = real_velocity
        self.moveit2.max_acceleration = real_accel
        self.moveit2.cartesian_avoid_collisions = True
        self.moveit2.cartesian_jump_threshold = 2.0

        # ---------------- Yayıncılar ----------------
        self.force_pub = self.create_publisher(WrenchStamped, "/harmony/cleaning/force", 10)
        self.robot_status_pub = self.create_publisher(String, "/harmony/robot_status", 10)
        # Senaryo olayları (arayüz açılır pencereleri bunu dinler); temizleme bitince
        # 'CLEANING_COMPLETE' yayınlanır.
        self.scenario_event_pub = self.create_publisher(String, "/harmony/scenario_event", 10)
        # RViz işaretçilerini algılama düğümü yayımlar (tek yayıncı); burada
        # yalnızca durum güncellemesi yayımlanır, algılama düğümü işaretçi
        # renklerini buna göre günceller.
        self.defect_status_pub = self.create_publisher(String, hd.TOPIC_DEFECT_STATUS, 10)

        # ---------------- Aboneler ----------------
        self.cmd_sub = self.create_subscription(
            String, "/harmony/cmd_input", self._cmd_cb, 10, callback_group=cbg)
        self.defect_sub = self.create_subscription(
            String, hd.TOPIC_DEFECT, self._defect_cb, 10, callback_group=cbg
        )

        # ---------------- Durum ----------------
        self.confirm_event = Event()
        self.stop_event = Event()
        self.cleaning_active = Event()

        self.defect_lock = Lock()
        self.scan_id: Optional[str] = None
        self.defects: List[Dict[str, Any]] = []
        self.last_cleaned_scan_id: Optional[str] = None
        # defect_id -> DETECTED / CLEANING / CLEANED (işaretçi renkleri için)
        self.defect_status: Dict[str, str] = {}

        self.force_timer = self.create_timer(1.0 / self.force_rate_hz, self._publish_force)

        self.worker = Thread(target=self._run, daemon=True)
        self.worker.start()

        self._publish_robot_status("IDLE", "IDLE", "CR node ready (waiting defect + CONFIRM)")
        self.get_logger().info(
            f"HarmonyCleaningMissionRunnerV5 (REAL) initialized | "
            f"group={realrobot.MOVE_GROUP_ARM} | planner={planner_id} | "
            f"vel={real_velocity} acc={real_accel} | order={hd.CLEANING_ORDER}"
        )

    def _publish_robot_status(self, state: str, mode: str, note: str, level: str = "INFO") -> None:
        payload = {
            "timestamp": _now_ros(self),
            "state": state,
            "mode": mode,
            "level": level,
            "note": note,
            "frame_id": self.fixed_frame,
        }
        out = String()
        out.data = json.dumps(payload)
        self.robot_status_pub.publish(out)

    def _publish_scenario_event(
            self, event: str, note: str = "", extra: Optional[dict] = None) -> None:
        """Senaryo olayı yayımlar (arayüz açılır pencereleri bunu dinler)."""
        payload = {
            "event": event,
            "note": note,
            "timestamp": _now_ros(self),
        }
        if extra:
            payload.update(extra)
        msg = String()
        msg.data = json.dumps(payload)
        self.scenario_event_pub.publish(msg)
        self.get_logger().info(f"Scenario event published: {event}")

    # ---------------- Girdi ayrıştırma ----------------
    def _extract_xyz(self, d: Dict[str, Any]) -> Optional[List[float]]:
        pos = None
        if isinstance(d.get("pos"), dict):
            pos = d["pos"]
        elif isinstance(d.get("position"), dict):
            pos = d["position"]
        elif all(k in d for k in ("x", "y", "z")):
            pos = d

        if not isinstance(pos, dict):
            return None

        try:
            return [float(pos["x"]), float(pos["y"]), float(pos["z"])]
        except Exception:
            return None

    # ---------------- Geri çağırmalar ----------------
    def _defect_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                return
        except Exception:
            self.get_logger().warning("Invalid JSON on /harmony/mock_perception/defect: "
                                      f"{msg.data[:200]}")
            return

        xyz = self._extract_xyz(data)
        if xyz is None:
            self.get_logger().warning("No x,y,z in defect payload")
            return

        defect_id = str(data.get("defect_id", "")).strip()

        with self.defect_lock:
            # Aynı kusur ikinci kez gelirse (yeniden yayın) kopyalama.
            if defect_id and any(d.get("defect_id") == defect_id for d in self.defects):
                return
            self.defects.append({
                "defect_id": defect_id,
                "defect_type": str(data.get("defect_type", "")),
                "position": {"x": xyz[0], "y": xyz[1], "z": xyz[2]},
                "raw": data,
            })
            n = len(self.defects)
            if defect_id:
                self.defect_status[defect_id] = hd.STATUS_DETECTED

        self._publish_robot_status("WAITING", "CR", f"Defect received: {n} total. Waiting CONFIRM")
        self.get_logger().info(f"Defect collected: {defect_id or '?'} ({n} total)")

    def _cmd_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            cmd = str(data.get("cmd", "")).upper().strip()
        except Exception:
            return

        # Temizleme sırasında CONFIRM gelirse yok say
        if cmd == "CONFIRM" and self.cleaning_active.is_set():
            return

        if cmd == "START":
            with self.defect_lock:
                self.defects.clear()
                self.defect_status.clear()
                self.scan_id = None
            self.get_logger().info("START received - cleared old defects")
        elif cmd == "CONFIRM":
            self.confirm_event.set()
        elif cmd == "WAITING":
            self._publish_robot_status("WAITING", "CR", "WAITING command received, staying in "
                                                        "WAITING")
        elif cmd == "STOP":
            self.stop_event.set()
            self.confirm_event.clear()
            self.cleaning_active.clear()
            self._publish_robot_status("IDLE", "IDLE", "STOP received, abort "
                                                       "cleaning", level="WARN")

    # ---------------- Kuvvet yayıncısı ----------------
    def _publish_force(self) -> None:
        if not self.cleaning_active.is_set():
            return

        self._publish_robot_status("CR_MODE", "CR", "Cleaning in progress")

        fz = random.uniform(5.0, 20.0)
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.tool_frame
        msg.wrench.force.z = fz
        self.force_pub.publish(msg)

    # ---------------- Hareket ----------------
    def move_to_pose(self, position: List[float], cartesian: bool = False) -> bool:
        """Gerçek robotta planlar ve çalıştırır."""
        quat = [0.0, 0.0, 0.0, 1.0]
        pos_f = [float(position[0]), float(position[1]), float(position[2])]

        trajectory = None
        try:
            trajectory = self.moveit2.plan(
                position=pos_f,
                quat_xyzw=quat,
                cartesian=cartesian,
                cartesian_fraction_threshold=self.cartesian_fraction_threshold,
            )
        except Exception as e:
            self.get_logger().warning(f"[CR] pose plan() hatası: {e}")

        if not trajectory:
            self.get_logger().warning("[CR] pose plan() başarısız → fallback: move_to_pose")
            try:
                self.moveit2.move_to_pose(
                    position=pos_f,
                    quat_xyzw=quat,
                    cartesian=cartesian,
                    cartesian_fraction_threshold=self.cartesian_fraction_threshold,
                )
                return bool(self.moveit2.wait_until_executed())
            except Exception as e:
                self.get_logger().warning(f"[CR] move_to_pose hatası: {e}")
                return False

        return self._execute(trajectory)

    def move_to_joint_config(self, joints: List[float], step_key: Optional[str] = None) -> bool:
        """
        Gerçek robotta planlar ve çalıştırır (isteğe bağlı yörünge önbelleğiyle).

        step_key verilirse yörünge kayıt/oynatma devreye girer: kayıtlı bir yörünge varsa
        planlama yapılmadan doğrudan oynatılır; yoksa planlanıp (kayıt kipinde) saklanır.
        """
        joints_f = [float(x) for x in joints]

        # Oynatma: kayıtlı yörünge varsa planlamadan doğrudan çalıştır.
        if step_key is not None and self.traj_store is not None:
            recorded = self.traj_store.get(step_key)
            if recorded is not None:
                if self._execute(recorded):
                    return True
                self.get_logger().warning(
                    f"[CR] Kayıtlı '{step_key}' çalıştırılamadı → yeniden planlanıyor."
                )

        trajectory = None
        try:
            trajectory = self.moveit2.plan(joint_positions=joints_f)
        except Exception as e:
            self.get_logger().warning(f"[CR] plan() hatası: {e}")

        if not trajectory:
            self.get_logger().warning("[CR] plan() başarısız → fallback: move_to_configuration")
            try:
                self.moveit2.move_to_configuration(joints_f)
                return bool(self.moveit2.wait_until_executed())
            except Exception as e:
                self.get_logger().warning(f"[CR] move_to_configuration hatası: {e}")
                return False

        # Kayıt kipi: yeni hesaplanan yörüngeyi sakla (save() ile dosyaya yazılır).
        if (
            step_key is not None
            and self.traj_store is not None
            and not self.traj_store.replay
        ):
            self.traj_store.record(step_key, trajectory)

        return self._execute(trajectory)

    def _execute(self, trajectory: JointTrajectory) -> bool:
        try:
            self.moveit2.execute(trajectory)
            return bool(self.moveit2.wait_until_executed())
        except Exception as e:
            self.get_logger().warning(f"[CR] execute hatası: {e}")
            return False

    # ---------------- Kusur durumu ----------------
    def _set_defect_status(self, defect_id: str, status: str) -> None:
        """Durumu yayımlar; işaretçi renklerini algılama düğümü günceller."""
        with self.defect_lock:
            self.defect_status[defect_id] = status

        msg = String()
        msg.data = hd.status_payload(defect_id, status)
        self.defect_status_pub.publish(msg)

    def move_with_retries(
            self, position: List[float], try_non_cartesian_first: bool = True) -> bool:
        for attempt in range(1, self.move_max_retries + 1):
            if self.stop_event.is_set():
                return False

            ok = False
            if try_non_cartesian_first:
                ok = self.move_to_pose(position, cartesian=False)
                if not ok:
                    ok = self.move_to_pose(position, cartesian=True)
            else:
                ok = self.move_to_pose(position, cartesian=True)
                if not ok:
                    ok = self.move_to_pose(position, cartesian=False)

            if ok:
                return True

            time.sleep(self.move_retry_sleep_s)

        return False

    # ---------------- Çalışan iş parçacığı ----------------
    def _ordered_defects(self, defects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Kusurları harmony_defects.CLEANING_ORDER sırasına dizer.

        Sıralama y'ye göre artandır (DEF-03 -> DEF-02 -> DEF-01 -> DEF-04); böylece
        doğrusal eksen tek yönde ilerler. Listede olmayan kusurlar sona eklenir.
        """
        rank = {did: i for i, did in enumerate(hd.CLEANING_ORDER)}
        return sorted(defects, key=lambda d: rank.get(d.get("defect_id", ""), len(rank)))

    def _clean_one(self, defect: Dict[str, Any]) -> bool:
        """Tek bir kusura gider ve orada cleaning_duration kadar bekler."""
        defect_id = str(defect.get("defect_id", ""))
        joints = hd.DEFECT_JOINTS.get(defect_id)

        if joints is None:
            self.get_logger().warning(
                f"[CR] {defect_id or '?'} için tanımlı eklem açısı yok, atlanıyor. "
                f"harmony_defects.DEFECT_JOINTS içine ekleyin."
            )
            return False

        pos = defect.get("position", {})
        self._publish_robot_status(
            "CR_MODE", "CR",
            f"Moving to {defect_id} ({defect.get('defect_type', '')}) "
            f"x={pos.get('x', 0):.3f} y={pos.get('y', 0):.3f} z={pos.get('z', 0):.3f}"
        )
        self._set_defect_status(defect_id, hd.STATUS_CLEANING)

        ok = self.move_to_joint_config(joints, step_key=defect_id)
        if not ok:
            # DETECTED'a geri dönmüyoruz: işaretçi kırmızı/turuncuya döner ve hata
            # "hiçbir şey olmamış" gibi görünür. FAILED ayrı renkte (mor) kalır.
            self.get_logger().error(
                f"[CR] {defect_id} hedefine GİDİLEMEDİ — MoveIt planlayamadı veya "
                f"execute başarısız. Eklem açıları: {[round(v, 4) for v in joints]}"
            )
            self._set_defect_status(defect_id, hd.STATUS_FAILED)
            return False

        t0 = time.time()
        while time.time() - t0 < self.cleaning_duration:
            if self.stop_event.is_set():
                break
            time.sleep(0.01)

        self._set_defect_status(defect_id, hd.STATUS_CLEANED)
        self._publish_robot_status("CR_MODE", "CR", f"{defect_id} cleaned")
        return True

    def _run(self) -> None:
        time.sleep(0.5)

        while rclpy.ok():
            if self.stop_event.is_set():
                time.sleep(0.1)
                continue

            if not self.confirm_event.wait(timeout=0.2):
                continue
            self.confirm_event.clear()

            with self.defect_lock:
                defects_copy = list(self.defects)
                self.defects = []
                self.scan_id = None

            if len(defects_copy) == 0:
                if self.skip_if_no_defects:
                    self._publish_robot_status("COMPLETE", "CR", "No defects to clean (complete)")
                    self._publish_robot_status("IDLE", "IDLE", "Back to IDLE")
                continue

            # NOT: Eskiden burada scan_id tabanlı bir "aynı tarama" koruması
            # vardı, ancak self.scan_id hiçbir zaman gerçek bir değere atanmadığı
            # için scan_key her turda sabit "no_scan_id" oluyordu; bu da ilk
            # turdan sonra TÜM turları engelliyordu (2. turda temizleme çalışmıyordu).
            # Çift-CONFIRM koruması zaten (a) temizlik sırasında gelen CONFIRM'in
            # cleaning_active ile yok sayılması ve (b) kusurlar tüketildikten
            # sonra self.defects boş olduğu için yukarıdaki len==0 -> skip dalı
            # ile sağlanıyor. Bu yüzden koruma kaldırıldı.
            ordered = self._ordered_defects(defects_copy)
            self._publish_robot_status(
                "CR_MODE", "CR",
                f"Cleaning started (n={len(ordered)}, order="
                f"{', '.join(d.get('defect_id', '?') for d in ordered)})"
            )
            self.cleaning_active.set()

            # Kusurlar arasında eve dönülmez; doğrudan sıradakine geçilir.
            cleaned = 0
            for defect in ordered:
                if self.stop_event.is_set():
                    break
                if self._clean_one(defect):
                    cleaned += 1
                elif not self.continue_on_move_failure:
                    self.get_logger().warning("[CR] Hareket başarısız, görev kesiliyor")
                    break

            self.cleaning_active.clear()

            if self.stop_event.is_set():
                self._publish_robot_status("IDLE", "IDLE", "Cleaning stopped", level="WARN")
                self.stop_event.clear()
                continue

            # Görev sonunda eve dön.
            self._publish_robot_status("CR_MODE", "CR", "All defects done. Returning HOME")
            self.move_to_joint_config(self.home_joints, step_key="home_end")
            # Kayıt kipindeysek bu turda hesaplanan yörüngeleri dosyaya yaz.
            if self.traj_store is not None and not self.traj_store.replay:
                self.traj_store.save()

            self._publish_robot_status(
                "COMPLETE", "CR", f"Cleaning completed ({cleaned}/{len(ordered)} defects)"
            )
            # Arayüz açılır penceresi: temizlenen nokta sayısıyla temizleme tamamlandı.
            self._publish_scenario_event(
                "CLEANING_COMPLETE",
                f"Temizlik tamamlandı ({cleaned}/{len(ordered)} defect)",
                extra={"cleaned": cleaned, "total": len(ordered)},
            )
            time.sleep(2.0)
            self._publish_robot_status("IDLE", "IDLE", "Back to IDLE, waiting defect + CONFIRM")


def main() -> None:
    rclpy.init()
    node = HarmonyCleaningMissionRunnerV5()
    exec_ = MultiThreadedExecutor(num_threads=6)
    exec_.add_node(node)
    exec_.add_node(node.moveit_node)
    try:
        exec_.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.moveit_node.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
