#!/usr/bin/env python3
"""
Kawasaki RS005L tek kol muayene düğümü.

Kawasaki'yi (AGV rayı world_to_agv + 6 kol eklemi) kendi kaplama bakış noktalarına
götürür ve SICK kamerasının verisini toplar. Kawasaki EKLEM UZAYINDA planlar
(+/-2*pi sınır kenarı normalleştirmesiyle) ve bir sonraki adıma geçmeyi ÖLÇÜLEN
varışa bağlar; çünkü AGV'si asenkron bir rosbridge bağlantısı üzerinden sürülür ve
JTC eylem sonucu, yavaş AGV fiziksel olarak varmadan çok önce gelir.

Ortak altyapı için bkz. inspection_base.InspectionNodeBase.
"""
import os
import re
import time
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.task import Future
from moveit_msgs.msg import PlanningScene
from trajectory_msgs.msg import JointTrajectory
from control_msgs.action import FollowJointTrajectory
from scipy.spatial.transform import Rotation

from book_pymoveit2 import MoveIt2 as MoveIt2Real

from ch07_multirobot_planner.inspection_base import InspectionNodeBase

# Kawasaki planlama grubunun ("real_kawasaki") eklemleri. Plan, bakış noktası başına
# bütün robotun çözümünü saklar; MoveIt2 nesnesi YALNIZCA bu eklemlerle kurulmalıdır
# (her biri /joint_states'te görünmedikçe eklem durumu önbelleği atılır).
KAWASAKI_JOINT_NAMES = [
    "world_to_agv", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
]


class KawasakiInspectionNode(InspectionNodeBase):
    # En yakın dal IK araması bu ikisine Kawasaki'de ihtiyaç duyar, UR'de hiçbirine
    # duymaz (varsayılanlarını oluşturdukları parametreler için bkz. InspectionNodeBase;
    # iki kol TEK bir başlatma parametresi sözlüğünü paylaşır, bu yüzden kol başına ayrım burada
    # durur).
    #
    # Kawasaki YEREL bir çözücü olan KDL ile çözer: eskiden önerilen iki üç tohumun hepsi
    # planlanan hedefin üzerindeydi; bu yüzden hepsi doğrudan ona yakınsıyor ve yürütücü
    # her bakış noktasında "1 distinct branch(es)" yazıyordu. Rastgele kol tohumları,
    # genel bir çözücünün içeride yapacağı şeydir; 24 rastgele tohumlu bir /compute_ik
    # denemesi aynı pozlar için 16-20 farklı çözüm döndürdü.
    DEFAULT_IK_RANDOM_SEEDS = 12
    # ...ve bu çözümler sonra atılıyordu, çünkü 7 serbestlik dereceli "real_kawasaki"
    # grubu KDL'nin world_to_agv'yi kaydırmasına izin veriyor: kawa_vp_005 için 19
    # çözümden yalnızca 1'i (kawa_vp_002 için 16'dan 3'ü) rayı planın koyduğu yerde
    # tutuyordu ve branch_max_rail_shift 0.0'dır. "real_kawasaki_arm", base_link'ten
    # başlayan aynı zincirdir ve üst eklemi (base_mount) SABİTTİR; ray grupta yoktur, kayamaz.
    DEFAULT_BRANCH_IK_GROUP = "real_kawasaki_arm"

    def __init__(self) -> None:
        super().__init__("kawasaki_inspection_node", robot_tag="kawasaki")

        # Kawasaki'ye özgü parametreler.
        self.declare_parameter("enable_kawasaki", True)
        # Bu ikisi multirobot_planner_node'daki aynı adlı değerlerle AYNI olmalıdır; yoksa
        # yürütücünün IK'sı planlayıcının çözdüğünden farklı bir pozu hedefler.
        self.declare_parameter("kawasaki_view_axis_correction_rpy",
                               [-1.5707963, -1.5707963, 0.0])
        self.declare_parameter("kawasaki_group_tip", "link6")
        self.declare_parameter("kawasaki_velocity", 0.007)
        self.declare_parameter("kawasaki_acceleration", 0.007)
        self.declare_parameter(
            "kawasaki_controller_action",
            "/kawasaki/kawasaki_controller/follow_joint_trajectory")
        # Veri toplama akışları.
        self.declare_parameter("sim_kawasaki_topic", "/sim/kawasaki/pointcloud")
        self.declare_parameter("real_kawasaki_topic", "/kawasaki/pointcloud")
        self.declare_parameter("sim_kawasaki_sensor_frame", "sim_kawasaki_camera_rgb_optic_frame")
        # Bulutun kendi başlık çerçevesi bulunamazsa kullanılan yedek çerçeve. HIL başlatma
        # dosyasının gerçek buluta damgaladığı KALİBRE edilmiş sick çerçevesiyle hizalıdır;
        # böylece anlık bir TF kaybı eski, kalibre edilmemiş camera_rgb_optic_frame'e dönmez.
        self.declare_parameter("real_kawasaki_sensor_frame", "kawasaki_sick_optical_frame")
        # Başlangıç / ana konumlar (0. indeks AGV rayı METRE, 1..6 kol eklemleri DERECE).
        self.declare_parameter("start_pose_kawasaki", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Son bakış noktasından sonra eve dön (UR'nin hep yaptığı gibi). Bu kolu eve
        # döndürmek AGV'ye de komut verir; eskiden isteğe bağlı olmasının sebebi budur.
        # Sonunda AGV'nin hareket etmemesi gereken bir koşu için return_home_kawasaki:=false verin.
        self.declare_parameter("return_home_kawasaki", True)
        # EV == BAŞLANGIÇ, bilerek; UR'nin aynası (onun ur_home_joint_positions ve
        # start_pose_ur değerleri aynı yedi değerdir). Eski varsayılan rayı 0.0 m'ye
        # koyuyordu; koşunun başladığı 1.0 m'den bir metre uzağa. Yani yalnızca eve dönüşü
        # açmak her koşuyu yavaş AGV'yi planın hiç seçmediği bir yere sürerek bitirirdi.
        # "Ev" AGV'nin başlangıcı değil de fiziksel istasyonu olmalıysa buradaki ray
        # elemanını yeniden 0.0 yapın.
        self.declare_parameter("kawasaki_home_joint_positions",
                               [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.arm_label = "Kawasaki"
        self.start_pose_param = "start_pose_kawasaki"
        self.return_home_param = "return_home_kawasaki"
        self.home_positions_param = "kawasaki_home_joint_positions"
        self.align_on_fresh = True  # Kawasaki'nin mevcut davranışını koru
        # UR'nin yolculuğu düzeltildiğinden beri kullandığı oynatma yöntemi: ayrı bir
        # hizalanma hamlesiyle oraya gitmek yerine kayıtlı yolun ilk noktasını ölçülen poza
        # taşı. _prepare_start önce yoldaki tam turları sarar; bu yüzden bu kolda taşıma
        # yalnızca küçük bir artığı karşılamak zorunda kalır ve fark daha büyükse
        # reddeder (açık yürüyüşe düşer).
        self.rebase_to_current = True

        # Veri toplama akışları: benzetim her zaman; gerçek yalnızca fiziksel hücreye bağlıyken.
        base = os.path.expanduser(self.get_parameter("output_base_dir").value)
        self.streams = [self._make_stream(
            "sim_kawasaki", "kawasaki",
            self.get_parameter("sim_kawasaki_topic").value,
            self.get_parameter("sim_kawasaki_sensor_frame").value,
            os.path.join(base, "sim_pcds", "kawasaki_data"))]
        if not self.only_sim:
            self.streams.append(self._make_stream(
                "real_kawasaki", "kawasaki",
                self.get_parameter("real_kawasaki_topic").value,
                self.get_parameter("real_kawasaki_sensor_frame").value,
                os.path.join(base, "real_pcds", "kawasaki_data")))
        self._subscribe_streams()

        # Kawasaki MoveIt2 (AGV rayı + 6 kol eklemi), YALNIZCA PLANLAMA.
        self.moveit = MoveIt2Real(
            node=self,
            joint_names=list(KAWASAKI_JOINT_NAMES),
            base_link_name=self.world_frame,
            end_effector_name="link6",
            group_name="real_kawasaki",
            callback_group=self._cb,
            # ur_inspection_node'daki notun aynısı: sarma burada yürütücüye aittir (önbellek
            # politika etiketi, kayıtlı yol sarma, ham son çare denemesi); bu yüzden MoveIt2'nin
            # kendi geçişi, işin iki kez yapılmaması için kapatılır.
            unwind_joint_goals=False,
        )
        # Taban sınıfta en yakın dal IK'sını etkinleştirir (aşağıdaki _ik_target'a bakın).
        self.ik_group_name = "real_kawasaki"
        self.moveit.max_velocity = self.get_parameter("kawasaki_velocity").value
        self.moveit.max_acceleration = self.get_parameter("kawasaki_acceleration").value
        self.moveit.allowed_planning_time = self.get_parameter("allowed_planning_time").value
        self.moveit.num_planning_attempts = self.get_parameter("num_planning_attempts").value
        self.moveit.planner_id = self.get_parameter("planner_id").value
        self.ctrl = ActionClient(
            self, FollowJointTrajectory,
            self.get_parameter("kawasaki_controller_action").value, callback_group=self._cb)

    # --- Kancalar ------------------------------------------------------- #
    def should_run(self) -> bool:
        only_ur, only_kawa = self._resolve_only_flags()
        if not self.get_parameter("enable_kawasaki").value:
            return False
        # Kawasaki, only_ur (only_kawa False iken) yoksa çalışır. İkisi de True ise zaten
        # sıfırlandı.
        return (not only_ur) or only_kawa

    def _use_pose_goal(self) -> bool:
        return False  # eklem uzayı + normalleştirme

    # --- En yakın dal IK ---------------------------------------------------- #
    # Kawasaki EKLEM UZAYINDA kalır: saklanan hedef birincil hedef olmaya devam eder ve
    # AGV rayı burada asla yeniden seçilmez. Buranın eklediği, AYNI kamera pozunun
    # DİĞER IK çözümleri arasında bir aramadır; böylece kol planın varsaydığı yerde
    # değilse (evden ilk bakış noktası, eve uğrama, atlanan bir bakış noktası, yeniden
    # hizalanan önbellekli bir yol) saklanan dal yerine en yakın dalı alabilir.
    # Poz her adayda aynıdır; dolayısıyla toplanan bulut değişemez.
    #
    # `vp['rotation']`, ÜRETİCİNİN görüş çerçevesidir (+Z görüş ekseni). Onu IK'nın
    # kullanabileceği bir şeye çevirmek, multirobot_planner_node'un _make_kawasaki_reach_fn
    # içinde kullandığı zincirin aynısını gerektirir; ikisi uyumlu kalmazsa kamera yanlış yöne
    # bakar:
    #   R_cam = R_view @ C          C = görüş ekseni düzeltmesi (+Z görüş -> kamera +X görüş)
    #   R_tip = R_cam @ R_off       (R_off, p_off) = link6'nın kamera çerçevesindeki pozu
    #   p_tip = pos + R_cam @ p_off
    # Kamera çerçevesi yerine grubun kendi ucu (link6) için çözmek bilinçli bir seçimdir:
    # MoveIt grup ucunun ötesindeki bir ik_link'i dikkate almaz; Kawasaki kamerasını bir
    # zamanlar ters yöne baktıran hata buydu.

    @staticmethod
    def _camera_to_tip() -> Tuple[np.ndarray, np.ndarray]:
        """
        (R_off, p_off): link6'nın camera_rgb_optic_frame içindeki pozu.

        Değerler doğrudan rs005l_macro.xacro'daki URDF sabitlerinden alınır (TF'den
        DEĞİL: öneksiz çerçeveler benzetim koşusunda TF'de yoktur ve buradaki sessiz bir
        arama hatası kamerayı yanlış yöne çevirirdi). Planlayıcıdaki
        _analytic_kawasaki_offset'in aynasıdır.

            link6 --joint7 (rpy 0 0 -1.96)--> link7
                  --camera_joint (xyz .0145 .0125 .185, rpy 0 -pi/2 pi/2)--> kamera
        """
        def rpy(r: float, p: float, y: float) -> np.ndarray:
            return Rotation.from_euler("xyz", [r, p, y]).as_matrix()
        R67 = rpy(0.0, 0.0, -1.96)
        R7c = rpy(0.0, -np.pi / 2.0, np.pi / 2.0)
        p7c = np.array([0.0145, 0.0125, 0.185])
        R6c = R67 @ R7c
        p6c = R67 @ p7c
        return R6c.T, -R6c.T @ p6c

    def _ik_target(self, vp: Dict[str, Any]) -> Optional[Tuple[List[float], np.ndarray, str]]:
        if vp.get("position") is None or vp.get("rotation") is None:
            return None
        C = Rotation.from_euler(
            "xyz", list(self.get_parameter("kawasaki_view_axis_correction_rpy").value)
        ).as_matrix()
        R_off, p_off = self._camera_to_tip()
        R_cam = np.asarray(vp["rotation"], dtype=float) @ C
        pos = np.asarray(vp["position"], dtype=float)
        R_tip = R_cam @ R_off
        p_tip = pos + R_cam @ p_off
        quat = Rotation.from_matrix(R_tip).as_quat()  # xyzw
        return [float(v) for v in p_tip], quat, self.get_parameter(
            "kawasaki_group_tip").value

    def _arm_link_names(self, scene: PlanningScene) -> List[str]:
        names = list(scene.allowed_collision_matrix.entry_names)
        kawa = re.compile(r"^link[1-6]$")
        return [n for n in names if kawa.match(n)]

    def _settle(self, send: Future, targets: Optional[Dict[str, float]], label: str) -> bool:
        # Kawasaki: ÖLÇÜLEN varışa göre ilerle (asenkron AGV, JTC sonucundan sonra yerleşir).
        if self._confirm_accepted(send, label):
            if targets:
                return self._wait_until_reached(targets, label)
            return True
        return False

    def _viewpoint_targets(
            self, traj: JointTrajectory, vp: Dict[str, Any]) -> Optional[Dict[str, float]]:
        return self._arrival_targets_from_traj(traj, vp)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = KawasakiInspectionNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    Thread(target=executor.spin, daemon=True).start()
    time.sleep(3.0)  # eylem sunucuları / TF / köprü ayağa kalksın

    try:
        node.run()  # MoveIt eylem çağrıları ana iş parçacığında
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
