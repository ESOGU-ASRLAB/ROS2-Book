#!/usr/bin/env python3
"""
UR10e tek kol muayene düğümü.

UR10e'yi (ray + 6 kol eklemi) kendi kaplama bakış noktalarına götürür ve SICK
kamerasının verisini toplar. UR, KARTEZYEN bir poz hedefine (planlayıcının IK
çözdüğü kamera optik çerçevesi) planlar; böylece MoveIt geçerli ve doğru yönelimli
bir eklem çözümünü yeniden türetir. Tamamlanmaya YALNIZCA denetleyicinin eylem
sonucuyla karar verilir; yürütücü tarafında ölçülen varış toleransı denetimi
yoktur. Bu sayede ray geri bildirimi bu kolun /joint_states konusunda görünmeyebilen
gerçek UR, bir tolerans denetimine asla takılmaz.

YÜRÜTME YOLU (ray düzeltmesi): UR, ham bir FollowJointTrajectory hedefini doğrudan
denetleyiciye GÖNDERMEZ. Bunun yerine MoveIt'in `execute_trajectory` eylemi
(move_group) üzerinden yürütür; bu, 110 döngüde kanıtlanmış book_pymoveit2
örneklerinin (ur_inspection_scenario / combined_joint_goal) kullandığı yöntemin AYNISIDIR.
move_group'un yürütme yöneticisi 7 eklemli yörüngenin tamamını (ray + 6 kol)
scaled_joint_trajectory_controller'a o kanıtlanmış sistemdeki gibi yönlendirir; rayın
sarsılmasına yol açan ham denetleyici kestirmesi kullanılmaz. Kawasaki ham denetleyici
gönderimini korur (kısıtı değişmedi).

Ortak altyapı için bkz. inspection_base.InspectionNodeBase.
"""
import os
import time
from threading import Thread
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.task import Future
from moveit_msgs.msg import PlanningScene
from trajectory_msgs.msg import JointTrajectory
from moveit_msgs.action import ExecuteTrajectory

from book_pymoveit2 import MoveIt2 as MoveIt2Real
from book_pymoveit2.robots import ur as ur_robot

from ch07_multirobot_planner.inspection_base import InspectionNodeBase


class URInspectionNode(InspectionNodeBase):
    def __init__(self) -> None:
        super().__init__("ur_inspection_node", robot_tag="ur")

        # UR'ye özgü parametreler.
        self.declare_parameter("ur_velocity", 0.1)
        self.declare_parameter("ur_acceleration", 0.1)
        self.declare_parameter(
            "ur_controller_action",
            "/scaled_joint_trajectory_controller/follow_joint_trajectory")
        # Poz hedefli planlama (YÖNELİM düzeltmesi): UR'yi planlayıcının IK çözdüğü kamera
        # optik çerçevesine planla; böylece MoveIt doğru yönelimli bir pozu yeniden çözer.
        self.declare_parameter("ur_use_pose_goal", True)
        self.declare_parameter("ur_tool_frame", "ur10e_depth_optical_frame")
        # Veri toplama akışları.
        self.declare_parameter("sim_ur_topic", "/sim/pointcloud")
        self.declare_parameter("real_ur_topic", "/sick_points")
        self.declare_parameter("sim_ur_sensor_frame", "sim_ur10e_depth_optical_frame")
        self.declare_parameter("real_ur_sensor_frame", "ur10e_sick_optical_frame")
        # Başlangıç / ana konumlar (0. indeks ray METRE, 1..6 kol eklemleri DERECE).
        self.declare_parameter("start_pose_ur", [1.0, 0.0, -90.0, 0.0, -90.0, 0.0, 0.0])
        self.declare_parameter("return_home_ur", True)
        self.declare_parameter("ur_home_joint_positions", [1.0, 0.0, -90.0, 0.0, -90.0, 0.0, 0.0])
        # NOT: SICK kamera çarpışma kabuğu (optik yüzde bir kutu, arkada bir kutu ve dört
        # yanın her birinde birer kutu) eskiden çalışma anında burada İLİŞTİRİLİYORDU,
        # çünkü sick_camera ağı gerçek gövdeyi olduğundan küçük gösteriyordu. Artık URDF'te
        # ur10e_sick_camera bağlantısı üzerinde altı <collision> kutusu olarak duruyor
        # (Universal_Robots_ROS2_Description/urdf/ur_macro.xacro); böylece move_group bu
        # düğüm bir şey eklemeden hem kayıt hem oynatma sırasında onu bilir. Geometri için
        # o bağlantının açıklamasına bakın.

        self.arm_label = "UR"
        self.start_pose_param = "start_pose_ur"
        self.return_home_param = "return_home_ur"
        self.home_positions_param = "ur_home_joint_positions"
        # UR: ayrı bir başlangıca hizalanma hareketi YERİNE her yörüngenin ilk noktasını
        # güncel ölçülen poza taşı (kanıtlanmış ur_inspection_scenario oynatması); rayın
        # yersiz ileri-geri hareketini durduran budur.
        self.rebase_to_current = True
        self.align_on_fresh = False  # (taşıma açıkken hizalama yolu artık kullanılmıyor)

        # Veri toplama akışları: benzetim her zaman; gerçek yalnızca fiziksel hücreye bağlıyken.
        base = os.path.expanduser(self.get_parameter("output_base_dir").value)
        self.streams = [self._make_stream(
            "sim_ur", "ur",
            self.get_parameter("sim_ur_topic").value,
            self.get_parameter("sim_ur_sensor_frame").value,
            os.path.join(base, "sim_pcds", "ur_data"))]
        if not self.only_sim:
            self.streams.append(self._make_stream(
                "real_ur", "ur",
                self.get_parameter("real_ur_topic").value,
                self.get_parameter("real_ur_sensor_frame").value,
                os.path.join(base, "real_pcds", "ur_data")))
        self._subscribe_streams()

        # UR10e MoveIt2 (ray + 6 kol eklemi), YALNIZCA PLANLAMA.
        self.moveit = MoveIt2Real(
            node=self,
            joint_names=ur_robot.joint_names(),
            base_link_name=self.world_frame,
            end_effector_name=ur_robot.end_effector_name(),
            group_name=ur_robot.MOVE_GROUP_ARM,
            callback_group=self._cb,
            # Bu yürütücü hedefleri MoveIt2'den daha fazla bağlam bilgisiyle kendisi sarar:
            # yörünge önbelleğini hedef politikasıyla etiketler, kayıtlı yolları sarar ve
            # sarılmış ve normalleştirilmiş denemelerden sonra bilerek HAM bir son çare
            # denemesi tutar. MoveIt2'nin de sarmasına izin vermek o ham yedeği ilk denemenin
            # bir kopyasına çevirir ve wrap_goals_to_current:=false değerinin hiçbir şeyi
            # kapatmamasına yol açardı.
            unwind_joint_goals=False,
        )
        # Taban sınıfın en yakın dal IK'sını etkinleştirir (poz hedefinin rastgele bir dal
        # seçmesine bırakmak yerine bu grup için ur_tool_frame'e /compute_ik çözer).
        self.ik_group_name = ur_robot.MOVE_GROUP_ARM
        self.moveit.max_velocity = self.get_parameter("ur_velocity").value
        self.moveit.max_acceleration = self.get_parameter("ur_acceleration").value
        self.moveit.allowed_planning_time = self.get_parameter("allowed_planning_time").value
        self.moveit.num_planning_attempts = self.get_parameter("num_planning_attempts").value
        self.moveit.planner_id = self.get_parameter("planner_id").value
        # UR, MoveIt'in execute_trajectory eylemiyle (move_group) yürütür; kanıtlanmış
        # robot.execute() yolu budur. move_group'un yürütme yöneticisi ray+kol yörüngesinin
        # tamamını scaled_joint_trajectory_controller'a 110 döngülük sistemdeki gibi yönlendirir.
        # Bu eylem DOĞRUDAN sürülür (ortak MultiThreadedExecutor üzerinden future tabanlı
        # kendi beklememizle); moveit.execute()/wait_until_executed() KULLANILMAZ, çünkü
        # onların içindeki rclpy.spin_once() yürütücümüzün wait set'ini bozar ve döndürme
        # iş parçacığını öldürür; bu da veri toplama aboneliklerini aç bırakır ("No point cloud").
        self.ctrl = ActionClient(
            self, ExecuteTrajectory, "execute_trajectory", callback_group=self._cb)

    # --- Kancalar ------------------------------------------------------- #
    def should_run(self) -> bool:
        only_ur, only_kawa = self._resolve_only_flags()
        # UR, only_kawasaki (only_ur False iken) yoksa çalışır. İkisi de True ise zaten sıfırlandı.
        return (not only_kawa) or only_ur

    def _use_pose_goal(self) -> bool:
        return bool(self.get_parameter("ur_use_pose_goal").value)

    def _pose_target_link(self) -> str:
        return self.get_parameter("ur_tool_frame").value

    def _arm_link_names(self, scene: PlanningScene) -> List[str]:
        names = list(scene.allowed_collision_matrix.entry_names)
        return [n for n in names if n.startswith("ur10e_")]

    def _dispatch(self, traj: JointTrajectory, label: str) -> Future:
        # KANITLANMIŞ YOL: move_group'un execute_trajectory eylemini ray+kol yörüngesinin
        # tamamıyla sür (robot.execute() karşılığı), ardından taban sınıfın future tabanlı
        # _await'iyle ortak MultiThreadedExecutor üzerinden bekle; spin_once yoktur, böylece
        # yürütücünün wait set'i bozulmaz ve veri toplama abonelikleri akmaya devam eder.
        # move_group önce hedefin başlangıç durumunu doğrular ve en yeni /joint_states
        # mesajı 1 sn'den eskiyse hedefi doğrudan REDDEDER ("couldn't receive full current
        # joint state within 1s"); denetleyici hedefi hiç görmez ve kol sessizce hareket etmez.
        # Her UR gönderimi (bakış noktaları, başlangıca hizalanma, başlangıç pozu, eve dönüş)
        # buradan geçer; bu yüzden gönderimden önceki tek bir tazelik beklemesi hepsini kapsar.
        self._wait_fresh_joint_state(label)
        goal = ExecuteTrajectory.Goal()
        goal.trajectory.joint_trajectory = traj
        return self.ctrl.send_goal_async(goal)

    def _settle(self, send: Future, targets: Optional[Dict[str, float]], label: str) -> bool:
        # UR YALNIZCA execute_trajectory sonucuna güvenir; ölçülen varış denetimi yoktur.
        if self._await(send, label):
            self.get_logger().info(f"{label} reached (execute_trajectory succeeded).")
            return True
        self.get_logger().warning(f"{label} motion did not report success; continuing anyway.")
        return False

    def _viewpoint_targets(
            self, traj: JointTrajectory, vp: Dict[str, Any]) -> Optional[Dict[str, float]]:
        return None  # yalnızca denetleyici; ölçülen varış denetimi yok


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = URInspectionNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    Thread(target=executor.spin, daemon=True).start()
    time.sleep(3.0)  # eylem sunucuları / TF ayağa kalksın

    try:
        node.run()  # MoveIt eylem çağrıları ana iş parçacığında
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
