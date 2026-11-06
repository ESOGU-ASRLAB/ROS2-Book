import copy
import threading
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningScene,
    PositionConstraint,
)
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetCartesianPath,
    GetMotionPlan,
    GetPlanningScene,
    GetPositionFK,
    GetPositionIK,
)
from rclpy.action import ActionClient
from rclpy.callback_groups import CallbackGroup
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.task import Future
from sensor_msgs.msg import JointState
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from std_msgs.msg import Header, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from book_pymoveit2.joint_wrap import (
    describe_changes,
    parse_joint_limits,
    path_travel,
    wrap_to_reference,
)
from book_pymoveit2.utils import enum_to_str


class MoveIt2State(Enum):
    """
    MoveIt2 arayüzünün o anki yürütme durumunu gösteren sayısal sabitler.

    - IDLE: Hiçbir hareket istenmiyor ya da yürütülmüyor.
    - REQUESTING: Yürütme istendi, ancak istek henüz kabul edilmedi.
    - EXECUTING: Yürütme istendi ve kabul edildi, henüz tamamlanmadı.
    """

    IDLE = 0
    REQUESTING = 1
    EXECUTING = 2


class MoveIt2:
    """
    Yörünge planlama ve yürütmeyi sağlayan MoveIt 2 Python arayüzü.

    Yürütme için robotun JointTrajectoryController kullanması gerekir.
    """

    def __init__(
        self,
        node: Node,
        joint_names: List[str],
        base_link_name: str,
        end_effector_name: str,
        group_name: str = "my_robot_cell",
        execute_via_moveit: bool = False,
        ignore_new_calls_while_executing: bool = False,
        callback_group: Optional[CallbackGroup] = None,
        follow_joint_trajectory_action_name: str = "DEPRECATED",
        use_move_group_action: bool = False,
        unwind_joint_goals: bool = True,
        unwind_limit_margin: float = 0.05,
        unwind_min_gain: float = 0.35,
    ) -> None:
        """
        `MoveIt2` arayüzünün bir örneğini oluşturur.

          - `node` - Arayüzün bağlı olduğu ROS 2 düğümü
          - `joint_names` - Robotun eklem adları listesi (URDF'den çıkarılabilir)
          - `base_link_name` - Robot taban bağlantısının adı
          - `end_effector_name` - Robot uç işlevcisinin adı
          - `group_name` - Robot kolunun planlama grubunun adı
          - [KULLANIMDAN KALKTI] `execute_via_moveit` - Yürütmeyi MoveGroup eylemiyle
            (MoveIt 2) yapan bayrak; aksi hâlde ayrı bir planlama servis istemcisiyle
            birlikte FollowJointTrajectory eylemi (denetleyici) kullanılır
          - `ignore_new_calls_while_executing` - Önceki yörünge yürütülürken yeni yörünge
            isteklerini yok sayan bayrak
          - `callback_group` - ROS 2 iletişimi (konular/servisler/eylemler) için
            isteğe bağlı geri çağırma grubu
          - [KULLANIMDAN KALKTI] `follow_joint_trajectory_action_name` - Denetleyicinin
            eylem sunucusunun adı
          - `use_move_group_action` - Yürütmeyi MoveGroup eylemiyle (MoveIt 2) yapan
            bayrak; aksi hâlde ayrı bir planlama servis istemcisiyle birlikte
            ExecuteTrajectory eylemi kullanılır
          - `unwind_joint_goals` - Her eklem hedefini, URDF sınırları içinde kalarak
            başlangıç durumuna en yakın 2*pi eşdeğeriyle yeniden ifade eder. Bkz.
            `set_joint_goal()`; açık bırakmak güvenlidir, çünkü yeniden yazılan hedef
            AYNI fiziksel konfigürasyondur.
          - `unwind_limit_margin` - Bir eşdeğer kabul edilirken her eklemin sınır
            kenarlarından uzak tutulan pay [radyan].
          - `unwind_min_gain` - Bir eklem yalnızca en az bu kadar radyan yol
            kazandırıyorsa yeniden yazılır (varsayılan ~20 derece).
        """
        self._node = node
        self._callback_group = callback_group

        # Kullanımdan kalkmış parametreleri denetle
        if execute_via_moveit:
            self._node.get_logger().warn(
                "Parameter `execute_via_moveit` is deprecated. Please use `use_move_group_action` "
                "instead."
            )
            use_move_group_action = True
        if follow_joint_trajectory_action_name != "DEPRECATED":
            self._node.get_logger().warn(
                "Parameter `follow_joint_trajectory_action_name` is deprecated. `MoveIt2` uses "
                "the `execute_trajectory` action instead."
            )

        self.__collision_object_publisher = self._node.create_publisher(
            CollisionObject, "collision_object", 10
        )
        self.__attached_collision_object_publisher = self._node.create_publisher(
            AttachedCollisionObject, "attached_collision_object", 10
        )

        self.__cancellation_pub = self._node.create_publisher(
            String, "trajectory_execution_event", 1
        )

        self.__joint_state_mutex = threading.Lock()
        self.__joint_state = None
        self.__new_joint_state_available = False
        self.__move_action_goal = self.__init_move_action_goal(
            frame_id=base_link_name,
            group_name=group_name,
            end_effector=end_effector_name,
        )

        # Yörüngelerin MoveGroup eylemiyle mi, yoksa ayrı ExecuteTrajectory eylemi
        # çağrılarak mı yürütüleceğini belirleyen bayrak
        # `move_to_pose()` ve `move_to_configuration()` için geçerlidir
        self.__use_move_group_action = use_move_group_action

        # Önceki hedef yürütülürken yeni bir hedef gönderilip gönderilemeyeceğini belirleyen bayrak
        self.__ignore_new_calls_while_executing = ignore_new_calls_while_executing

        # Sonradan kullanılacak ek değişkenleri sakla
        self.__joint_names = joint_names
        self.__base_link_name = base_link_name
        self.__end_effector_name = end_effector_name
        self.__group_name = group_name

        # Güncel hareket isteklerini ve yürütmeyi izleyen iç durumlar
        self.__is_motion_requested = False
        self.__is_executing = False
        self.motion_suceeded = False
        self.__execution_goal_handle = None
        self.__last_error_code = None
        self.__execution_mutex = threading.Lock()

        # Güncel eklem durumları için abone oluştur
        self._node.create_subscription(
            msg_type=JointState,
            topic="joint_states",
            callback=self.__joint_state_callback,
            qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
            callback_group=self._callback_group,
        )

        # 2*pi eklem hedefi sarma. URDF /robot_description konusundan alınır;
        # robot_state_publisher bu konuyu kalıcı (latched) yayımladığı için, düğüm çok
        # sonra başlasa bile TRANSIENT_LOCAL bir abone onu alır. Hiçbir şey onu beklemez:
        # gelene kadar sınır tablosu boştur ve her sarma etkisiz kalır; yani modelsiz
        # bir koşu, bu özellik eklenmeden önceki gibi davranır.
        self.__unwind_joint_goals = bool(unwind_joint_goals)
        self.__unwind_limit_margin = float(unwind_limit_margin)
        self.__unwind_min_gain = float(unwind_min_gain)
        self.__urdf_xml = None
        self.__joint_limits = None
        self.__unwind_warned = False
        # Abonelik KOŞULSUZ açılır, yalnızca sarma açıkken değil: `joint_limits`, başka
        # özelliklerin de okuduğu herkese açık bir özelliktir (nearest_branch_ik ona,
        # doğrusal rayı dönel eklemden ayırmak ve rastgele tohumları gerçek sınırlar
        # içinde çekmek için ihtiyaç duyar). Aboneliği unwind_joint_goals'a bağlamak,
        # sarmayı kendisi yapan her çağıranda (üç muayene yürütücüsünün hepsi) tabloyu
        # boş bırakıyordu; yol puanlaması da sessizce ray METRELERİNİ bilek RADYANLARINA
        # ekliyordu. Sarmanın kendisi yine unwind_joint_goal() içindeki bayrağa bağlıdır;
        # bu satır yalnızca modeli erişilebilir kılar.
        self._node.create_subscription(
            msg_type=String,
            topic="robot_description",
            callback=self.__robot_description_callback,
            qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
            callback_group=self._callback_group,
        )

        # Hareket (move) eylemi için eylem istemcisi oluştur
        self.__move_action_client = ActionClient(
            node=self._node,
            action_type=MoveGroup,
            action_name="move_action",
            goal_service_qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
            result_service_qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=5,
            ),
            cancel_service_qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=5,
            ),
            feedback_sub_qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
            status_sub_qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
            callback_group=self._callback_group,
        )

        # Planlama için ayrı bir servis istemcisi de oluştur
        self._plan_kinematic_path_service = self._node.create_client(
            srv_type=GetMotionPlan,
            srv_name="plan_kinematic_path",
            qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
            callback_group=callback_group,
        )
        self.__kinematic_path_request = GetMotionPlan.Request()

        # Kartezyen planlama için ayrı bir servis istemcisi oluştur
        self._plan_cartesian_path_service = self._node.create_client(
            srv_type=GetCartesianPath,
            srv_name="compute_cartesian_path",
            qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
            callback_group=callback_group,
        )
        self.__cartesian_path_request = GetCartesianPath.Request()

        # Yörünge yürütme için eylem istemcisi oluştur
        self._execute_trajectory_action_client = ActionClient(
            node=self._node,
            action_type=ExecuteTrajectory,
            action_name="execute_trajectory",
            goal_service_qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
            result_service_qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=5,
            ),
            cancel_service_qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=5,
            ),
            feedback_sub_qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
            status_sub_qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
            callback_group=self._callback_group,
        )

        # Planlama sahnesini almak için servis istemcisi oluştur
        self._get_planning_scene_service = self._node.create_client(
            srv_type=GetPlanningScene,
            srv_name="get_planning_scene",
            qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
            callback_group=callback_group,
        )
        self.__planning_scene = None
        self.__old_planning_scene = None
        self.__old_allowed_collision_matrix = None

        # Planlama sahnesini uygulamak için servis istemcisi oluştur
        self._apply_planning_scene_service = self._node.create_client(
            srv_type=ApplyPlanningScene,
            srv_name="apply_planning_scene",
            qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
            callback_group=callback_group,
        )

    # --- Yürütme durumunu yoklayan fonksiyonlar ---
    def query_state(self) -> MoveIt2State:
        with self.__execution_mutex:
            if self.__is_motion_requested:
                return MoveIt2State.REQUESTING
            elif self.__is_executing:
                return MoveIt2State.EXECUTING
            else:
                return MoveIt2State.IDLE

    def cancel_execution(self) -> None:
        if self.query_state() != MoveIt2State.EXECUTING:
            self._node.get_logger().warn("Attempted to cancel without active goal.")
            return None

        cancel_string = String()
        cancel_string.data = "stop"
        self.__cancellation_pub.publish(cancel_string)

    def get_execution_future(self) -> Optional[Future]:
        if self.query_state() != MoveIt2State.EXECUTING:
            self._node.get_logger().warn("Need active goal for future.")
            return None

        return self.__execution_goal_handle.get_result_async()

    def get_last_execution_error_code(self) -> Optional[MoveItErrorCodes]:
        return self.__last_error_code

    # ---

    def move_to_pose(
        self,
        pose: Optional[Union[PoseStamped, Pose]] = None,
        position: Optional[Union[Point, Tuple[float, float, float]]] = None,
        quat_xyzw: Optional[
            Union[Quaternion, Tuple[float, float, float, float]]
        ] = None,
        target_link: Optional[str] = None,
        frame_id: Optional[str] = None,
        tolerance_position: float = 0.001,
        tolerance_orientation: float = 0.001,
        weight_position: float = 1.0,
        cartesian: bool = False,
        weight_orientation: float = 1.0,
        cartesian_max_step: float = 0.0025,
        cartesian_fraction_threshold: float = 0.0,
    ) -> None:
        """
        Önceden ayarlanmış hedeflere göre hareketi planlar ve yürütür.

        İsteğe bağlı argümanlar verilirse, çağrı sırasında hedefi tanımlamak için
        içeride `set_pose_goal()` kullanılır.
        """
        if isinstance(pose, PoseStamped):
            pose_stamped = pose
        elif isinstance(pose, Pose):
            pose_stamped = PoseStamped(
                header=Header(
                    stamp=self._node.get_clock().now().to_msg(),
                    frame_id=(
                        frame_id if frame_id is not None else self.__base_link_name
                    ),
                ),
                pose=pose,
            )
        else:
            if not isinstance(position, Point):
                position = Point(
                    x=float(position[0]), y=float(position[1]), z=float(position[2])
                )
            if not isinstance(quat_xyzw, Quaternion):
                quat_xyzw = Quaternion(
                    x=float(quat_xyzw[0]),
                    y=float(quat_xyzw[1]),
                    z=float(quat_xyzw[2]),
                    w=float(quat_xyzw[3]),
                )
            pose_stamped = PoseStamped(
                header=Header(
                    stamp=self._node.get_clock().now().to_msg(),
                    frame_id=(
                        frame_id if frame_id is not None else self.__base_link_name
                    ),
                ),
                pose=Pose(position=position, orientation=quat_xyzw),
            )

        if self.__use_move_group_action and not cartesian:
            if self.__ignore_new_calls_while_executing and (
                self.__is_motion_requested or self.__is_executing
            ):
                self._node.get_logger().warn(
                    "Controller is already following a trajectory. Skipping motion."
                )
                return

            # Hedefi ayarla
            self.set_pose_goal(
                position=pose_stamped.pose.position,
                quat_xyzw=pose_stamped.pose.orientation,
                frame_id=pose_stamped.header.frame_id,
                target_link=target_link,
                tolerance_position=tolerance_position,
                tolerance_orientation=tolerance_orientation,
                weight_position=weight_position,
                weight_orientation=weight_orientation,
            )
            # Başlangıç durumu olarak güncel durumu tanımla
            if self.joint_state is not None:
                self.__move_action_goal.request.start_state.joint_state = (
                    self.joint_state
                )
            # Hedefi sunucuya gönder (asenkron) - planlama ve yürütme birlikte
            self._send_goal_async_move_action()
            # Önceki bütün hedef kısıtlarını temizle
            self.clear_goal_constraints()
            self.clear_path_constraints()

        else:
            # MoveIt 2 ile planla, ardından doğrudan denetleyiciyle yürüt
            self.execute(
                self.plan(
                    position=pose_stamped.pose.position,
                    quat_xyzw=pose_stamped.pose.orientation,
                    frame_id=pose_stamped.header.frame_id,
                    target_link=target_link,
                    tolerance_position=tolerance_position,
                    tolerance_orientation=tolerance_orientation,
                    weight_position=weight_position,
                    weight_orientation=weight_orientation,
                    cartesian=cartesian,
                    max_step=cartesian_max_step,
                    cartesian_fraction_threshold=cartesian_fraction_threshold,
                )
            )

    def move_to_configuration(
        self,
        joint_positions: List[float],
        joint_names: Optional[List[str]] = None,
        tolerance: float = 0.001,
        weight: float = 1.0,
    ) -> None:
        """
        Önceden ayarlanmış hedeflere göre hareketi planlar ve yürütür.

        İsteğe bağlı argümanlar verilirse, çağrı sırasında hedefi tanımlamak için
        içeride `set_joint_goal()` kullanılır.
        """
        if self.__use_move_group_action:
            if self.__ignore_new_calls_while_executing and (
                self.__is_motion_requested or self.__is_executing
            ):
                self._node.get_logger().warn(
                    "Controller is already following a trajectory. Skipping motion."
                )
                return

            # Hedefi ayarla
            self.set_joint_goal(
                joint_positions=joint_positions,
                joint_names=joint_names,
                tolerance=tolerance,
                weight=weight,
            )
            # Başlangıç durumu olarak güncel durumu tanımla
            if self.joint_state is not None:
                self.__move_action_goal.request.start_state.joint_state = (
                    self.joint_state
                )
            # Hedefi sunucuya gönder (asenkron) - planlama ve yürütme birlikte
            self._send_goal_async_move_action()
            # Önceki bütün hedef kısıtlarını temizle
            self.clear_goal_constraints()
            self.clear_path_constraints()

        else:
            # MoveIt 2 ile planla, ardından doğrudan denetleyiciyle yürüt
            self.execute(
                self.plan(
                    joint_positions=joint_positions,
                    joint_names=joint_names,
                    tolerance_joint_position=tolerance,
                    weight_joint_position=weight,
                )
            )

    def plan(
        self,
        pose: Optional[Union[PoseStamped, Pose]] = None,
        position: Optional[Union[Point, Tuple[float, float, float]]] = None,
        quat_xyzw: Optional[
            Union[Quaternion, Tuple[float, float, float, float]]
        ] = None,
        joint_positions: Optional[List[float]] = None,
        joint_names: Optional[List[str]] = None,
        frame_id: Optional[str] = None,
        target_link: Optional[str] = None,
        tolerance_position: float = 0.001,
        tolerance_orientation: Union[float, Tuple[float, float, float]] = 0.001,
        tolerance_joint_position: float = 0.001,
        weight_position: float = 1.0,
        weight_orientation: float = 1.0,
        weight_joint_position: float = 1.0,
        start_joint_state: Optional[Union[JointState, List[float]]] = None,
        cartesian: bool = False,
        max_step: float = 0.0025,
        cartesian_fraction_threshold: float = 0.0,
    ) -> Optional[JointTrajectory]:
        """plan_async'i çağırır ve sonucu (future) bekler."""
        future = self.plan_async(
            **{
                key: value
                for key, value in locals().items()
                if key not in ["self", "cartesian_fraction_threshold"]
            }
        )

        if future is None:
            return None

        while not future.done():
            self._spin_or_sleep(timeout_sec=1.0)

        return self.get_trajectory(
            future,
            cartesian=cartesian,
            cartesian_fraction_threshold=cartesian_fraction_threshold,
        )

    def plan_async(
        self,
        pose: Optional[Union[PoseStamped, Pose]] = None,
        position: Optional[Union[Point, Tuple[float, float, float]]] = None,
        quat_xyzw: Optional[
            Union[Quaternion, Tuple[float, float, float, float]]
        ] = None,
        joint_positions: Optional[List[float]] = None,
        joint_names: Optional[List[str]] = None,
        frame_id: Optional[str] = None,
        target_link: Optional[str] = None,
        tolerance_position: float = 0.001,
        tolerance_orientation: Union[float, Tuple[float, float, float]] = 0.001,
        tolerance_joint_position: float = 0.001,
        weight_position: float = 1.0,
        weight_orientation: float = 1.0,
        weight_joint_position: float = 1.0,
        start_joint_state: Optional[Union[JointState, List[float]]] = None,
        cartesian: bool = False,
        max_step: float = 0.0025,
    ) -> Optional[Future]:
        """
        Önceden ayarlanmış hedeflere göre hareket planlar.

        İsteğe bağlı argümanlar verilirse, çağrı sırasında hedefi tanımlamak için
        içeride `set_position_goal()`, `set_orientation_goal()` ya da `set_joint_goal()`
        kullanılır. Zaman aşımı süresi içinde yörünge bulunamazsa `None` döner. Güncel
        konumdan farklı bir konumdan planlamak için isteğe bağlı `start_` argümanı
        verilebilir.
        """
        pose_stamped = None
        if pose is not None:
            if isinstance(pose, PoseStamped):
                pose_stamped = pose
            elif isinstance(pose, Pose):
                pose_stamped = PoseStamped(
                    header=Header(
                        stamp=self._node.get_clock().now().to_msg(),
                        frame_id=(
                            frame_id if frame_id is not None else self.__base_link_name
                        ),
                    ),
                    pose=pose,
                )

            self.set_position_goal(
                position=pose_stamped.pose.position,
                frame_id=pose_stamped.header.frame_id,
                target_link=target_link,
                tolerance=tolerance_position,
                weight=weight_position,
            )
            self.set_orientation_goal(
                quat_xyzw=pose_stamped.pose.orientation,
                frame_id=pose_stamped.header.frame_id,
                target_link=target_link,
                tolerance=tolerance_orientation,
                weight=weight_orientation,
            )
        else:
            if position is not None:
                if not isinstance(position, Point):
                    position = Point(
                        x=float(position[0]), y=float(position[1]), z=float(position[2])
                    )

                self.set_position_goal(
                    position=position,
                    frame_id=frame_id,
                    target_link=target_link,
                    tolerance=tolerance_position,
                    weight=weight_position,
                )

            if quat_xyzw is not None:
                if not isinstance(quat_xyzw, Quaternion):
                    quat_xyzw = Quaternion(
                        x=float(quat_xyzw[0]),
                        y=float(quat_xyzw[1]),
                        z=float(quat_xyzw[2]),
                        w=float(quat_xyzw[3]),
                    )

                self.set_orientation_goal(
                    quat_xyzw=quat_xyzw,
                    frame_id=frame_id,
                    target_link=target_link,
                    tolerance=tolerance_orientation,
                    weight=weight_orientation,
                )

        if joint_positions is not None:
            self.set_joint_goal(
                joint_positions=joint_positions,
                joint_names=joint_names,
                tolerance=tolerance_joint_position,
                weight=weight_joint_position,
                # Planın gerçekten başladığı duruma göre sar. Çağıran bir başlangıç durumu
                # vermediyse burada hâlâ None'dır ve set_joint_goal güncel ölçülen duruma
                # düşer; aşağıdaki döngünün zaten çözeceği değer de tam olarak budur.
                reference=start_joint_state,
            )
        # Planın başlangıç durumunu tanımla (varsayılan: güncel durum)
        while start_joint_state is None:
            self._node._logger.warn(message="Joint states are not available yet!")
            if self.__joint_state is not None:
                start_joint_state = self.__joint_state
                break
            else:
                self._spin_or_sleep(timeout_sec=1.0)
        self._node._logger.info(message="Joint states are available now")

        # Yörüngeyi servis çağrısıyla asenkron planla
        if cartesian:
            future = self._plan_cartesian_path(
                max_step=max_step,
                frame_id=(
                    pose_stamped.header.frame_id
                    if pose_stamped is not None
                    else frame_id
                ),
            )
        else:
            # Servisi kullan
            future = self._plan_kinematic_path()

        # Önceki bütün hedef kısıtlarını temizle
        self.clear_goal_constraints()
        self.clear_path_constraints()

        return future

    def get_trajectory(
        self,
        future: Future,
        cartesian: bool = False,
        cartesian_fraction_threshold: float = 0.0,
    ) -> Optional[JointTrajectory]:
        """
        plan_async'in döndürdüğü future'dan yörüngeyi alır.

        Future tamamlanmış ve planlama başarılıysa yörüngeyi, değilse None döndürür.

        Kartezyen planlarda, yolun tamamlanan oranı `cartesian_fraction_threshold`
        değerinden küçükse plan reddedilir.
        """
        if not future.done():
            self._node.get_logger().warn(
                "Cannot get trajectory because future is not done."
            )
            return None

        res = future.result()

        # Kartezyen
        if cartesian:
            if MoveItErrorCodes.SUCCESS == res.error_code.val:
                if res.fraction >= cartesian_fraction_threshold:
                    return res.solution.joint_trajectory
                else:
                    self._node.get_logger().warn(
                        f"Planning failed! Cartesian planner completed {res.fraction} "
                        "of the trajectory, less than the threshold "
                        f"{cartesian_fraction_threshold}."
                    )
                    return None
            else:
                self._node.get_logger().warn(
                    "Planning failed! Error code: "
                    f"{enum_to_str(MoveItErrorCodes, res.error_code.val)}"
                )
                return None

        # Değilse kinematik
        res = res.motion_plan_response
        if MoveItErrorCodes.SUCCESS == res.error_code.val:
            return res.trajectory.joint_trajectory
        else:
            self._node.get_logger().warn(
                f"Planning failed! Error code: {enum_to_str(MoveItErrorCodes, res.error_code.val)}"
            )
            return None

    def execute(self, joint_trajectory: JointTrajectory) -> None:
        """joint_trajectory'yi doğrudan denetleyiciyle iletişim kurarak yürütür."""
        if self.__ignore_new_calls_while_executing and (
            self.__is_motion_requested or self.__is_executing
        ):
            self._node.get_logger().warn(
                "Controller is already following a trajectory. Skipping motion."
            )
            return

        execute_trajectory_goal = init_execute_trajectory_goal(
            joint_trajectory=joint_trajectory
        )

        if execute_trajectory_goal is None:
            self._node.get_logger().warn(
                "Cannot execute motion because the provided/planned trajectory is invalid."
            )
            return

        self._send_goal_async_execute_trajectory(goal=execute_trajectory_goal)

    def wait_until_executed(self) -> bool:
        """Önceden istenen hareket başarıyla ya da hatayla sonuçlanana kadar bekler."""
        if not self.__is_motion_requested:
            self._node.get_logger().warn(
                "Cannot wait until motion is executed (no motion is in progress)."
            )
            return False

        while self.__is_motion_requested or self.__is_executing:
            self._spin_or_sleep(timeout_sec=1.0)

        return self.motion_suceeded

    def reset_controller(
        self,
        joint_state: Union[JointState, List[float]],
    ) -> None:
        """
        Sahte bir eklem yörüngesi göndererek denetleyiciyi verilen `joint_state`'e sıfırlar.

        Eklemlerin anında sıfırlanabildiği benzetim robotlarında işe yarar.
        """
        if not isinstance(joint_state, JointState):
            joint_state = init_joint_state(
                joint_names=self.__joint_names,
                joint_positions=joint_state,
            )
        joint_trajectory = init_dummy_joint_trajectory_from_state(joint_state)
        execute_trajectory_goal = init_execute_trajectory_goal(
            joint_trajectory=joint_trajectory
        )

        self._send_goal_async_execute_trajectory(goal=execute_trajectory_goal)

    def set_pose_goal(
        self,
        pose: Optional[Union[PoseStamped, Pose]] = None,
        position: Optional[Union[Point, Tuple[float, float, float]]] = None,
        quat_xyzw: Optional[
            Union[Quaternion, Tuple[float, float, float, float]]
        ] = None,
        frame_id: Optional[str] = None,
        target_link: Optional[str] = None,
        tolerance_position: float = 0.001,
        tolerance_orientation: Union[float, Tuple[float, float, float]] = 0.001,
        weight_position: float = 1.0,
        weight_orientation: float = 1.0,
    ) -> None:
        """`set_position_goal()` ile `set_orientation_goal()` fonksiyonlarının birleşimi."""
        if (pose is None) and (position is None or quat_xyzw is None):
            raise ValueError(
                "Either `pose` or `position` and `quat_xyzw` must be specified!"
            )

        if isinstance(pose, PoseStamped):
            pose_stamped = pose
        elif isinstance(pose, Pose):
            pose_stamped = PoseStamped(
                header=Header(
                    stamp=self._node.get_clock().now().to_msg(),
                    frame_id=(
                        frame_id if frame_id is not None else self.__base_link_name
                    ),
                ),
                pose=pose,
            )
        else:
            if not isinstance(position, Point):
                position = Point(
                    x=float(position[0]), y=float(position[1]), z=float(position[2])
                )
            if not isinstance(quat_xyzw, Quaternion):
                quat_xyzw = Quaternion(
                    x=float(quat_xyzw[0]),
                    y=float(quat_xyzw[1]),
                    z=float(quat_xyzw[2]),
                    w=float(quat_xyzw[3]),
                )
            pose_stamped = PoseStamped(
                header=Header(
                    stamp=self._node.get_clock().now().to_msg(),
                    frame_id=(
                        frame_id if frame_id is not None else self.__base_link_name
                    ),
                ),
                pose=Pose(position=position, orientation=quat_xyzw),
            )

        self.set_position_goal(
            position=pose_stamped.pose.position,
            frame_id=pose_stamped.header.frame_id,
            target_link=target_link,
            tolerance=tolerance_position,
            weight=weight_position,
        )
        self.set_orientation_goal(
            quat_xyzw=pose_stamped.pose.orientation,
            frame_id=pose_stamped.header.frame_id,
            target_link=target_link,
            tolerance=tolerance_orientation,
            weight=weight_orientation,
        )

    def create_position_constraint(
        self,
        position: Union[Point, Tuple[float, float, float]],
        frame_id: Optional[str] = None,
        target_link: Optional[str] = None,
        tolerance: float = 0.001,
        weight: float = 1.0,
    ) -> PositionConstraint:
        """
        `target_link` için `frame_id`'ye göre Kartezyen konum kısıtı oluşturur.

          - `frame_id` varsayılan olarak taban bağlantısıdır
          - `target_link` varsayılan olarak uç işlevcidir
        """
        # Yeni konum kısıtı oluştur
        constraint = PositionConstraint()

        # Referans çerçeveyi ve hedef bağlantıyı tanımla
        constraint.header.frame_id = (
            frame_id if frame_id is not None else self.__base_link_name
        )
        constraint.link_name = (
            target_link if target_link is not None else self.__end_effector_name
        )

        # Hedef konumu tanımla
        constraint.constraint_region.primitive_poses.append(Pose())
        if isinstance(position, Point):
            constraint.constraint_region.primitive_poses[0].position = position
        else:
            constraint.constraint_region.primitive_poses[0].position.x = float(
                position[0]
            )
            constraint.constraint_region.primitive_poses[0].position.y = float(
                position[1]
            )
            constraint.constraint_region.primitive_poses[0].position.z = float(
                position[2]
            )

        # Hedef bölgeyi, yarıçapı toleransa eşit bir küre olarak tanımla
        constraint.constraint_region.primitives.append(SolidPrimitive())
        constraint.constraint_region.primitives[0].type = 2  # Küre
        constraint.constraint_region.primitives[0].dimensions = [tolerance]

        # Kısıtın ağırlığını ayarla
        constraint.weight = weight

        return constraint

    def set_position_goal(
        self,
        position: Union[Point, Tuple[float, float, float]],
        frame_id: Optional[str] = None,
        target_link: Optional[str] = None,
        tolerance: float = 0.001,
        weight: float = 1.0,
    ) -> None:
        """
        `target_link` için `frame_id`'ye göre Kartezyen konum hedefi ayarlar.

          - `frame_id` varsayılan olarak taban bağlantısıdır
          - `target_link` varsayılan olarak uç işlevcidir
        """
        constraint = self.create_position_constraint(
            position=position,
            frame_id=frame_id,
            target_link=target_link,
            tolerance=tolerance,
            weight=weight,
        )

        # Diğer kısıtlara ekle
        self.__move_action_goal.request.goal_constraints[
            -1
        ].position_constraints.append(constraint)

    def create_orientation_constraint(
        self,
        quat_xyzw: Union[Quaternion, Tuple[float, float, float, float]],
        frame_id: Optional[str] = None,
        target_link: Optional[str] = None,
        tolerance: Union[float, Tuple[float, float, float]] = 0.001,
        weight: float = 1.0,
        parameterization: int = 0,  # 0: Euler açıları, 1: Dönme vektörü
    ) -> OrientationConstraint:
        """
        `target_link` için `frame_id`'ye göre Kartezyen yönelim kısıtı oluşturur.

          - `frame_id` varsayılan olarak taban bağlantısıdır
          - `target_link` varsayılan olarak uç işlevcidir
        """
        # Yeni yönelim kısıtı oluştur
        constraint = OrientationConstraint()

        # Referans çerçeveyi ve hedef bağlantıyı tanımla
        constraint.header.frame_id = (
            frame_id if frame_id is not None else self.__base_link_name
        )
        constraint.link_name = (
            target_link if target_link is not None else self.__end_effector_name
        )

        # Hedef yönelimi tanımla
        if isinstance(quat_xyzw, Quaternion):
            constraint.orientation = quat_xyzw
        else:
            constraint.orientation.x = float(quat_xyzw[0])
            constraint.orientation.y = float(quat_xyzw[1])
            constraint.orientation.z = float(quat_xyzw[2])
            constraint.orientation.w = float(quat_xyzw[3])

        # Toleransları tanımla
        if type(tolerance) == float:
            tolerance_xyz = (tolerance, tolerance, tolerance)
        else:
            tolerance_xyz = tolerance
        constraint.absolute_x_axis_tolerance = tolerance_xyz[0]
        constraint.absolute_y_axis_tolerance = tolerance_xyz[1]
        constraint.absolute_z_axis_tolerance = tolerance_xyz[2]

        # Parametrelemeyi tanımla (toleransın nasıl yorumlanacağı)
        constraint.parameterization = parameterization

        # Kısıtın ağırlığını ayarla
        constraint.weight = weight

        return constraint

    def set_orientation_goal(
        self,
        quat_xyzw: Union[Quaternion, Tuple[float, float, float, float]],
        frame_id: Optional[str] = None,
        target_link: Optional[str] = None,
        tolerance: Union[float, Tuple[float, float, float]] = 0.001,
        weight: float = 1.0,
        parameterization: int = 0,  # 0: Euler açıları, 1: Dönme vektörü
    ) -> None:
        """
        `target_link` için `frame_id`'ye göre Kartezyen yönelim hedefi ayarlar.

          - `frame_id` varsayılan olarak taban bağlantısıdır
          - `target_link` varsayılan olarak uç işlevcidir
        """
        constraint = self.create_orientation_constraint(
            quat_xyzw=quat_xyzw,
            frame_id=frame_id,
            target_link=target_link,
            tolerance=tolerance,
            weight=weight,
            parameterization=parameterization,
        )

        # Diğer kısıtlara ekle
        self.__move_action_goal.request.goal_constraints[
            -1
        ].orientation_constraints.append(constraint)

    def create_joint_constraints(
        self,
        joint_positions: List[float],
        joint_names: Optional[List[str]] = None,
        tolerance: float = 0.001,
        weight: float = 1.0,
    ) -> List[JointConstraint]:
        """
        Eklem uzayı kısıtları oluşturur.

        `joint_names` verilirse `joint_positions` belirli eklemler için herhangi bir
        sırada tanımlanabilir. Verilmezse yapıcıya geçirilen ilk **n** eklem kullanılır;
        **n**, `joint_positions` listesinin uzunluğudur.
        """
        constraints = []

        # Belirtilmemişse varsayılan eklem adlarını kullan
        if joint_names is None:
            joint_names = self.__joint_names

        for i in range(len(joint_positions)):
            # Her eklem için yeni bir kısıt oluştur
            constraint = JointConstraint()

            # Eklem adını tanımla
            constraint.joint_name = joint_names[i]

            # Hedef eklem konumunu tanımla
            constraint.position = joint_positions[i]

            # Toleransları tanımla
            constraint.tolerance_above = tolerance
            constraint.tolerance_below = tolerance

            # Kısıtın ağırlığını ayarla
            constraint.weight = weight

            constraints.append(constraint)

        return constraints

    def set_joint_goal(
        self,
        joint_positions: List[float],
        joint_names: Optional[List[str]] = None,
        tolerance: float = 0.001,
        weight: float = 1.0,
        reference: Optional[Union[JointState, List[float]]] = None,
    ) -> None:
        """
        Eklem uzayı hedefi ayarlar.

        `joint_names` verilirse `joint_positions` belirli eklemler için herhangi bir
        sırada tanımlanabilir. Verilmezse yapıcıya geçirilen ilk **n** eklem kullanılır;
        **n**, `joint_positions` listesinin uzunluğudur.

        `unwind_joint_goals` ile kapatılmadıkça hedef önce `reference`'a (varsayılan:
        kolun güncel ölçülen durumu) en yakın 2*pi eşdeğeriyle yeniden ifade edilir.
        Eklem uzayındaki her hareket buradan geçer (`move_to_configuration()` ve
        `plan_async()` ikisi de buraya varır); bu yüzden sarmanın yeri burasıdır.
        Kolun varacağı pozu neden değiştiremeyeceği için bkz. `unwind_joint_goal()`.
        """
        constraints = self.create_joint_constraints(
            joint_positions=self.unwind_joint_goal(
                joint_positions, joint_names=joint_names, reference=reference
            ),
            joint_names=joint_names,
            tolerance=tolerance,
            weight=weight,
        )

        # Diğer kısıtlara ekle
        self.__move_action_goal.request.goal_constraints[-1].joint_constraints.extend(
            constraints
        )

    def clear_goal_constraints(self) -> None:
        """
        Önceden ayarlanmış bütün hedef kısıtlarını temizler.

        Bu fonksiyon her `plan_kinematic_path()` çağrısından sonra kendiliğinden çağrılır.
        """
        self.__move_action_goal.request.goal_constraints = [Constraints()]

    def create_new_goal_constraint(self) -> None:
        """
        İstekle birlikte gönderilecek yeni bir hedef kısıtları kümesi oluşturur.

        Bundan sonra `set_joint_goal()`, `set_pose_goal()` ve benzerleriyle ayarlanan
        her hedef bu yeni kısıt kümesine eklenir.
        """
        self.__move_action_goal.request.goal_constraints.append(Constraints())

    def set_path_joint_constraint(
        self,
        joint_positions: List[float],
        joint_names: Optional[List[str]] = None,
        tolerance: float = 0.001,
        weight: float = 1.0,
    ) -> None:
        """
        Eklem uzayı yol kısıtları ayarlar.

        `joint_names` verilirse `joint_positions` belirli eklemler için herhangi bir
        sırada tanımlanabilir. Verilmezse yapıcıya geçirilen ilk **n** eklem kullanılır;
        **n**, `joint_positions` listesinin uzunluğudur.
        """
        constraints = self.create_joint_constraints(
            joint_positions=joint_positions,
            joint_names=joint_names,
            tolerance=tolerance,
            weight=weight,
        )

        # Diğer kısıtlara ekle
        self.__move_action_goal.request.path_constraints.joint_constraints.extend(
            constraints
        )

    def set_path_position_constraint(
        self,
        position: Union[Point, Tuple[float, float, float]],
        frame_id: Optional[str] = None,
        target_link: Optional[str] = None,
        tolerance: float = 0.001,
        weight: float = 1.0,
    ) -> None:
        """
        `target_link` için `frame_id`'ye göre Kartezyen konum yol kısıtı ayarlar.

          - `frame_id` varsayılan olarak taban bağlantısıdır
          - `target_link` varsayılan olarak uç işlevcidir
        """
        constraint = self.create_position_constraint(
            position=position,
            frame_id=frame_id,
            target_link=target_link,
            tolerance=tolerance,
            weight=weight,
        )

        # Diğer kısıtlara ekle
        self.__move_action_goal.request.path_constraints.position_constraints.append(
            constraint
        )

    def set_path_orientation_constraint(
        self,
        quat_xyzw: Union[Quaternion, Tuple[float, float, float, float]],
        frame_id: Optional[str] = None,
        target_link: Optional[str] = None,
        tolerance: Union[float, Tuple[float, float, float]] = 0.001,
        weight: float = 1.0,
        parameterization: int = 0,  # 0: Euler açıları, 1: Dönme vektörü
    ) -> None:
        """
        `target_link` için `frame_id`'ye göre Kartezyen yönelim yol kısıtı ayarlar.

          - `frame_id` varsayılan olarak taban bağlantısıdır
          - `target_link` varsayılan olarak uç işlevcidir
        """
        constraint = self.create_orientation_constraint(
            quat_xyzw=quat_xyzw,
            frame_id=frame_id,
            target_link=target_link,
            tolerance=tolerance,
            weight=weight,
            parameterization=parameterization,
        )

        # Diğer kısıtlara ekle
        self.__move_action_goal.request.path_constraints.orientation_constraints.append(
            constraint
        )

    def clear_path_constraints(self) -> None:
        """
        Önceden ayarlanmış bütün yol kısıtlarını temizler.

        Bu fonksiyon her `plan_kinematic_path()` çağrısından sonra kendiliğinden çağrılır.
        """
        self.__move_action_goal.request.path_constraints = Constraints()

    def compute_fk(
        self,
        joint_state: Optional[Union[JointState, List[float]]] = None,
        fk_link_names: Optional[List[str]] = None,
    ) -> Optional[Union[PoseStamped, List[PoseStamped]]]:
        """compute_fk_async'i çağırır ve sonucu (future) bekler."""
        future = self.compute_fk_async(
            **{key: value for key, value in locals().items() if key != "self"}
        )

        if future is None:
            return None

        while not future.done():
            self._spin_or_sleep(timeout_sec=1.0)

        return self.get_compute_fk_result(future, fk_link_names=fk_link_names)

    def get_compute_fk_result(
        self,
        future: Future,
        fk_link_names: Optional[List[str]] = None,
    ) -> Optional[Union[PoseStamped, List[PoseStamped]]]:
        """
        compute_fk_async'in döndürdüğü future'dan pozları alır.

        Future tamamlanmış ve başarılıysa pozları, değilse None döndürür.
        """
        if not future.done():
            self._node.get_logger().warn(
                "Cannot get FK result because future is not done."
            )
            return None

        res = future.result()

        if MoveItErrorCodes.SUCCESS == res.error_code.val:
            if fk_link_names is None:
                return res.pose_stamped[0]
            else:
                return res.pose_stamped
        else:
            self._node.get_logger().warn(
                f"FK computation failed! Error code: {res.error_code.val}."
            )
            return None

    def compute_fk_async(
        self,
        joint_state: Optional[Union[JointState, List[float]]] = None,
        fk_link_names: Optional[List[str]] = None,
    ) -> Optional[Future]:
        """
        Verilen `joint_state` için bütün `fk_link_names` bağlantılarının ileri kinematiği.

          - `fk_link_names` varsayılan olarak uç işlevcidir
          - `joint_state` varsayılan olarak güncel eklem durumudur
        """
        if not hasattr(self, "__compute_fk_client"):
            self.__init_compute_fk()

        if fk_link_names is None:
            self.__compute_fk_req.fk_link_names = [self.__end_effector_name]
        else:
            self.__compute_fk_req.fk_link_names = fk_link_names

        if joint_state is not None:
            if isinstance(joint_state, JointState):
                self.__compute_fk_req.robot_state.joint_state = joint_state
            else:
                self.__compute_fk_req.robot_state.joint_state = init_joint_state(
                    joint_names=self.__joint_names,
                    joint_positions=joint_state,
                )
        elif self.joint_state is not None:
            self.__compute_fk_req.robot_state.joint_state = self.joint_state

        stamp = self._node.get_clock().now().to_msg()
        self.__compute_fk_req.header.stamp = stamp
        self.__compute_fk_client.wait_for_service(timeout_sec=3.0)
        if not self.__compute_fk_client.service_is_ready():
            self._node.get_logger().warn(
                f"Service '{self.__compute_fk_client.srv_name}' is not yet available. Better luck "
                "next time!"
            )
            return None

        return self.__compute_fk_client.call_async(self.__compute_fk_req)

    def compute_ik(
        self,
        position: Union[Point, Tuple[float, float, float]],
        quat_xyzw: Union[Quaternion, Tuple[float, float, float, float]],
        start_joint_state: Optional[Union[JointState, List[float]]] = None,
        constraints: Optional[Constraints] = None,
        wait_for_server_timeout_sec: Optional[float] = 1.0,
        ik_link_name: Optional[str] = None,
        ik_timeout: Optional[float] = None,
        avoid_collisions: Optional[bool] = None,
        group_name: Optional[str] = None,
    ) -> Optional[JointState]:
        """
        compute_ik_async'i çağırır ve sonucu (future) bekler.

        UYARI: Bu fonksiyon düğümü çağıran iş parçacığından döndürür (spin). Düğümünüz
        zaten bir yürütücü (executor) altında çalışıyorsa compute_ik_async() kullanıp
        future'ı kendiniz bekleyin; nearest_branch_ik() tam olarak bunu yapar.
        """
        future = self.compute_ik_async(
            **{key: value for key, value in locals().items() if key != "self"}
        )

        if future is None:
            return None

        while not future.done():
            self._spin_or_sleep(timeout_sec=1.0)

        return self.get_compute_ik_result(future)

    def get_compute_ik_result(
        self,
        future: Future,
    ) -> Optional[JointState]:
        """
        compute_ik_async'in döndürdüğü future'dan eklem durumlarını alır.

        Future tamamlanmış ve başarılıysa eklem durumlarını, değilse None döndürür.
        """
        if not future.done():
            self._node.get_logger().warn(
                "Cannot get IK result because future is not done."
            )
            return None

        res = future.result()

        if MoveItErrorCodes.SUCCESS == res.error_code.val:
            return res.solution.joint_state
        else:
            self._node.get_logger().warn(
                "IK computation failed! Error code: "
                f"{enum_to_str(MoveItErrorCodes, res.error_code.val)}"
            )
            return None

    def compute_ik_async(
        self,
        position: Union[Point, Tuple[float, float, float]],
        quat_xyzw: Union[Quaternion, Tuple[float, float, float, float]],
        start_joint_state: Optional[Union[JointState, List[float]]] = None,
        constraints: Optional[Constraints] = None,
        wait_for_server_timeout_sec: Optional[float] = 1.0,
        ik_link_name: Optional[str] = None,
        ik_timeout: Optional[float] = None,
        avoid_collisions: Optional[bool] = None,
        group_name: Optional[str] = None,
    ) -> Optional[Future]:
        """
        Verilen poz için ters kinematik hesaplar.

        Arama uzayının başlangıcını belirtmek için `start_joint_state` verilebilir. Ayrıca
        hesaplanan IK'ya `constraints` uygulanabilir.
          - `start_joint_state` varsayılan olarak güncel eklem durumudur.
          - `constraints` varsayılan olarak None'dır.
          - `ik_link_name` grubun varsayılan ucu yerine BU bağlantı için çözer. Bu
            olmadan, ucun ötesine takılı bir algılayıcı (örneğin bir kamera optik
            çerçevesi) hedef poza yerleştirilemez: MoveIt oraya flanşı koyar. Varsayılan
            grubun kendi ucudur, yani bu argüman eklenmeden önceki davranış.
          - `ik_timeout` IK çözücünün harcayabileceği süre [s]. Varsayılan, çözücünün
            kinematics.yaml dosyasındaki değeridir.
          - `avoid_collisions` bu çağrı için varsayılanı (True) değiştirir. False SAF
            KİNEMATİK erişilebilirliği ölçer; yalnızca tanı içindir, kol çarpışma
            içinde olabilir.
          - `group_name` bu nesnenin grubundan farklı bir grupta çözer. Planladığınız grup,
            çözücünün OYNATMAMASI gereken fazladan bir eksen (ray) içeriyorsa işe yarar:
            o ekseni içermeyen bir grupta çözmek kaymayı imkânsız kılar.

        İstek nesnesi çağrılar arasında yeniden kullanılır; bu yüzden isteğe bağlı her
        alan HER çağrıda yazılır. None vermek, önceki çağrının değerini bırakmak yerine
        varsayılanı geri yükler.
        """
        if not hasattr(self, "__compute_ik_client"):
            self.__init_compute_ik()

        self.__compute_ik_req.ik_request.ik_link_name = (
            ik_link_name if ik_link_name else ""
        )
        self.__compute_ik_req.ik_request.group_name = (
            group_name if group_name else self.__group_name
        )
        self.__compute_ik_req.ik_request.avoid_collisions = (
            True if avoid_collisions is None else bool(avoid_collisions)
        )
        if ik_timeout is None:
            self.__compute_ik_req.ik_request.timeout.sec = 0
            self.__compute_ik_req.ik_request.timeout.nanosec = 0
        else:
            self.__compute_ik_req.ik_request.timeout.sec = int(ik_timeout)
            self.__compute_ik_req.ik_request.timeout.nanosec = int(
                (float(ik_timeout) - int(ik_timeout)) * 1e9
            )

        if isinstance(position, Point):
            self.__compute_ik_req.ik_request.pose_stamped.pose.position = position
        else:
            self.__compute_ik_req.ik_request.pose_stamped.pose.position.x = float(
                position[0]
            )
            self.__compute_ik_req.ik_request.pose_stamped.pose.position.y = float(
                position[1]
            )
            self.__compute_ik_req.ik_request.pose_stamped.pose.position.z = float(
                position[2]
            )
        if isinstance(quat_xyzw, Quaternion):
            self.__compute_ik_req.ik_request.pose_stamped.pose.orientation = quat_xyzw
        else:
            self.__compute_ik_req.ik_request.pose_stamped.pose.orientation.x = float(
                quat_xyzw[0]
            )
            self.__compute_ik_req.ik_request.pose_stamped.pose.orientation.y = float(
                quat_xyzw[1]
            )
            self.__compute_ik_req.ik_request.pose_stamped.pose.orientation.z = float(
                quat_xyzw[2]
            )
            self.__compute_ik_req.ik_request.pose_stamped.pose.orientation.w = float(
                quat_xyzw[3]
            )

        if start_joint_state is not None:
            if isinstance(start_joint_state, JointState):
                self.__compute_ik_req.ik_request.robot_state.joint_state = (
                    start_joint_state
                )
            else:
                self.__compute_ik_req.ik_request.robot_state.joint_state = (
                    init_joint_state(
                        joint_names=self.__joint_names,
                        joint_positions=start_joint_state,
                    )
                )
        elif self.joint_state is not None:
            self.__compute_ik_req.ik_request.robot_state.joint_state = self.joint_state

        if constraints is not None:
            self.__compute_ik_req.ik_request.constraints = constraints

        stamp = self._node.get_clock().now().to_msg()
        self.__compute_ik_req.ik_request.pose_stamped.header.stamp = stamp
        self.__compute_ik_client.wait_for_service(timeout_sec=3.0)
        if not self.__compute_ik_client.wait_for_service(
            timeout_sec=wait_for_server_timeout_sec
        ):
            self._node.get_logger().warn(
                f"Service '{self.__compute_ik_client.srv_name}' is not yet available. Better luck "
                "next time!"
            )
            return None

        return self.__compute_ik_client.call_async(self.__compute_ik_req)

    def reset_new_joint_state_checker(self) -> None:
        """Yeni eklem durumu denetleyicisini sıfırlar."""
        self.__joint_state_mutex.acquire()
        self.__new_joint_state_available = False
        self.__joint_state_mutex.release()

    def force_reset_executing_state(self) -> None:
        """
        Yürütmeyi engelleyen iç durumları zorla sıfırlar.

        `ignore_new_calls_while_executing` kullanılırken geçerlidir. Bu fonksiyon
        yalnızca çok az sayıdaki uç durumda gerekir; neredeyse hiç kullanılmamalıdır.
        """
        self.__execution_mutex.acquire()
        self.__is_motion_requested = False
        self.__is_executing = False
        self.__execution_mutex.release()

    def add_collision_primitive(
        self,
        id: str,
        primitive_type: int,
        dimensions: Tuple[float, float, float],
        pose: Optional[Union[PoseStamped, Pose]] = None,
        position: Optional[Union[Point, Tuple[float, float, float]]] = None,
        quat_xyzw: Optional[
            Union[Quaternion, Tuple[float, float, float, float]]
        ] = None,
        frame_id: Optional[str] = None,
        operation: int = CollisionObject.ADD,
    ) -> None:
        """
        Boyutlarıyla belirtilen temel geometrili bir çarpışma nesnesi ekler.

        `primitive_type` şunlardan biri olabilir:
            - `SolidPrimitive.BOX`
            - `SolidPrimitive.SPHERE`
            - `SolidPrimitive.CYLINDER`
            - `SolidPrimitive.CONE`
        """
        if (pose is None) and (position is None or quat_xyzw is None):
            raise ValueError(
                "Either `pose` or `position` and `quat_xyzw` must be specified!"
            )

        if isinstance(pose, PoseStamped):
            pose_stamped = pose
        elif isinstance(pose, Pose):
            pose_stamped = PoseStamped(
                header=Header(
                    stamp=self._node.get_clock().now().to_msg(),
                    frame_id=(
                        frame_id if frame_id is not None else self.__base_link_name
                    ),
                ),
                pose=pose,
            )
        else:
            if not isinstance(position, Point):
                position = Point(
                    x=float(position[0]), y=float(position[1]), z=float(position[2])
                )
            if not isinstance(quat_xyzw, Quaternion):
                quat_xyzw = Quaternion(
                    x=float(quat_xyzw[0]),
                    y=float(quat_xyzw[1]),
                    z=float(quat_xyzw[2]),
                    w=float(quat_xyzw[3]),
                )
            pose_stamped = PoseStamped(
                header=Header(
                    stamp=self._node.get_clock().now().to_msg(),
                    frame_id=(
                        frame_id if frame_id is not None else self.__base_link_name
                    ),
                ),
                pose=Pose(position=position, orientation=quat_xyzw),
            )

        msg = CollisionObject(
            header=pose_stamped.header,
            id=id,
            operation=operation,
            pose=pose_stamped.pose,
        )

        msg.primitives.append(
            SolidPrimitive(type=primitive_type, dimensions=dimensions)
        )

        self.__collision_object_publisher.publish(msg)

    def add_collision_box(
        self,
        id: str,
        size: Tuple[float, float, float],
        pose: Optional[Union[PoseStamped, Pose]] = None,
        position: Optional[Union[Point, Tuple[float, float, float]]] = None,
        quat_xyzw: Optional[
            Union[Quaternion, Tuple[float, float, float, float]]
        ] = None,
        frame_id: Optional[str] = None,
        operation: int = CollisionObject.ADD,
    ) -> None:
        """Boyutlarıyla belirtilen kutu geometrili bir çarpışma nesnesi ekler."""
        assert len(size) == 3, "Invalid size of the box!"

        self.add_collision_primitive(
            id=id,
            primitive_type=SolidPrimitive.BOX,
            dimensions=size,
            pose=pose,
            position=position,
            quat_xyzw=quat_xyzw,
            frame_id=frame_id,
            operation=operation,
        )

    def add_collision_sphere(
        self,
        id: str,
        radius: float,
        pose: Optional[Union[PoseStamped, Pose]] = None,
        position: Optional[Union[Point, Tuple[float, float, float]]] = None,
        quat_xyzw: Optional[
            Union[Quaternion, Tuple[float, float, float, float]]
        ] = None,
        frame_id: Optional[str] = None,
        operation: int = CollisionObject.ADD,
    ) -> None:
        """Yarıçapıyla belirtilen küre geometrili bir çarpışma nesnesi ekler."""
        if quat_xyzw is None:
            quat_xyzw = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        self.add_collision_primitive(
            id=id,
            primitive_type=SolidPrimitive.SPHERE,
            dimensions=[
                radius,
            ],
            pose=pose,
            position=position,
            quat_xyzw=quat_xyzw,
            frame_id=frame_id,
            operation=operation,
        )

    def add_collision_cylinder(
        self,
        id: str,
        height: float,
        radius: float,
        pose: Optional[Union[PoseStamped, Pose]] = None,
        position: Optional[Union[Point, Tuple[float, float, float]]] = None,
        quat_xyzw: Optional[
            Union[Quaternion, Tuple[float, float, float, float]]
        ] = None,
        frame_id: Optional[str] = None,
        operation: int = CollisionObject.ADD,
    ) -> None:
        """Yüksekliği ve yarıçapıyla belirtilen silindir geometrili bir çarpışma nesnesi ekler."""
        self.add_collision_primitive(
            id=id,
            primitive_type=SolidPrimitive.CYLINDER,
            dimensions=[height, radius],
            pose=pose,
            position=position,
            quat_xyzw=quat_xyzw,
            frame_id=frame_id,
            operation=operation,
        )

    def add_collision_cone(
        self,
        id: str,
        height: float,
        radius: float,
        pose: Optional[Union[PoseStamped, Pose]] = None,
        position: Optional[Union[Point, Tuple[float, float, float]]] = None,
        quat_xyzw: Optional[
            Union[Quaternion, Tuple[float, float, float, float]]
        ] = None,
        frame_id: Optional[str] = None,
        operation: int = CollisionObject.ADD,
    ) -> None:
        """Yüksekliği ve yarıçapıyla belirtilen koni geometrili bir çarpışma nesnesi ekler."""
        self.add_collision_primitive(
            id=id,
            primitive_type=SolidPrimitive.CONE,
            dimensions=[height, radius],
            pose=pose,
            position=position,
            quat_xyzw=quat_xyzw,
            frame_id=frame_id,
            operation=operation,
        )

    def add_collision_mesh(
        self,
        filepath: Optional[str],
        id: str,
        pose: Optional[Union[PoseStamped, Pose]] = None,
        position: Optional[Union[Point, Tuple[float, float, float]]] = None,
        quat_xyzw: Optional[
            Union[Quaternion, Tuple[float, float, float, float]]
        ] = None,
        frame_id: Optional[str] = None,
        operation: int = CollisionObject.ADD,
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        mesh: Optional[Any] = None,
    ) -> None:
        """
        Ağ (mesh) geometrili bir çarpışma nesnesi ekler.

        `filepath` ya da `mesh` argümanlarından biri verilmelidir.
        Not: Bu fonksiyon için 'trimesh' Python modülünün kurulu olması gerekir.
        """
        # Ağı yükle
        try:
            import trimesh
        except ImportError as err:
            raise ImportError(
                "Python module 'trimesh' not found! Please install it manually in order "
                "to add collision objects into the MoveIt 2 planning scene."
            ) from err

        # Parametreleri denetle
        if (pose is None) and (position is None or quat_xyzw is None):
            raise ValueError(
                "Either `pose` or `position` and `quat_xyzw` must be specified!"
            )
        if (filepath is None and mesh is None) or (
            filepath is not None and mesh is not None
        ):
            raise ValueError("Exactly one of `filepath` or `mesh` must be specified!")
        if mesh is not None and not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("`mesh` must be an instance of `trimesh.Trimesh`!")

        if isinstance(pose, PoseStamped):
            pose_stamped = pose
        elif isinstance(pose, Pose):
            pose_stamped = PoseStamped(
                header=Header(
                    stamp=self._node.get_clock().now().to_msg(),
                    frame_id=(
                        frame_id if frame_id is not None else self.__base_link_name
                    ),
                ),
                pose=pose,
            )
        else:
            if not isinstance(position, Point):
                position = Point(
                    x=float(position[0]), y=float(position[1]), z=float(position[2])
                )
            if not isinstance(quat_xyzw, Quaternion):
                quat_xyzw = Quaternion(
                    x=float(quat_xyzw[0]),
                    y=float(quat_xyzw[1]),
                    z=float(quat_xyzw[2]),
                    w=float(quat_xyzw[3]),
                )
            pose_stamped = PoseStamped(
                header=Header(
                    stamp=self._node.get_clock().now().to_msg(),
                    frame_id=(
                        frame_id if frame_id is not None else self.__base_link_name
                    ),
                ),
                pose=Pose(position=position, orientation=quat_xyzw),
            )

        msg = CollisionObject(
            header=pose_stamped.header,
            id=id,
            operation=operation,
            pose=pose_stamped.pose,
        )

        if filepath is not None:
            mesh = trimesh.load(filepath)

        # Ağı ölçekle
        if isinstance(scale, float):
            scale = (scale, scale, scale)
        if not (scale[0] == scale[1] == scale[2] == 1.0):
            # Ağ parametre olarak verildiyse, özgün nesneyi dönüştürmemek için
            # bir kopyasını al.
            if filepath is None:
                mesh = mesh.copy()
            # Ağı dönüştür
            transform = np.eye(4)
            np.fill_diagonal(transform, scale)
            mesh.apply_transform(transform)

        # trimesh yüzlerini uint32'ye çevir (ROS 2 MeshTriangle gerekliliği)
        faces_uint32 = np.array(mesh.faces, dtype=np.uint32)

        msg.meshes.append(
            Mesh(
                triangles=[
                    MeshTriangle(vertex_indices=np.array(
                        [faces_uint32[i, 0], faces_uint32[i, 1], faces_uint32[i, 2]],
                        dtype=np.uint32))
                    for i in range(len(faces_uint32))
                ],
                vertices=[
                    Point(x=float(vert[0]), y=float(vert[1]), z=float(vert[2]))
                    for vert in mesh.vertices
                ],
            )
        )

        self.__collision_object_publisher.publish(msg)

    def remove_collision_object(self, id: str) -> None:
        """`id` ile belirtilen çarpışma nesnesini kaldırır."""
        msg = CollisionObject()
        msg.id = id
        msg.operation = CollisionObject.REMOVE
        msg.header.stamp = self._node.get_clock().now().to_msg()
        self.__collision_object_publisher.publish(msg)

    def remove_collision_mesh(self, id: str) -> None:
        """
        `id` ile belirtilen çarpışma ağını kaldırır.

        `remove_collision_object()` ile aynıdır.
        """
        self.remove_collision_object(id)

    def attach_collision_object(
        self,
        id: str,
        link_name: Optional[str] = None,
        touch_links: List[str] = [],
        weight: float = 0.0,
    ) -> None:
        """Çarpışma nesnesini robota iliştirir."""
        if link_name is None:
            link_name = self.__end_effector_name

        msg = AttachedCollisionObject(
            object=CollisionObject(id=id, operation=CollisionObject.ADD)
        )
        msg.link_name = link_name
        msg.touch_links = touch_links
        msg.weight = weight

        self.__attached_collision_object_publisher.publish(msg)

    def detach_collision_object(self, id: int) -> None:
        """Çarpışma nesnesini robottan ayırır."""
        msg = AttachedCollisionObject(
            object=CollisionObject(id=id, operation=CollisionObject.REMOVE)
        )
        self.__attached_collision_object_publisher.publish(msg)

    def detach_all_collision_objects(self) -> None:
        """Çarpışma nesnesini robottan ayırır."""
        msg = AttachedCollisionObject(
            object=CollisionObject(operation=CollisionObject.REMOVE)
        )
        self.__attached_collision_object_publisher.publish(msg)

    def move_collision(
        self,
        id: str,
        position: Union[Point, Tuple[float, float, float]],
        quat_xyzw: Union[Quaternion, Tuple[float, float, float, float]],
        frame_id: Optional[str] = None,
    ) -> None:
        """`id` ile belirtilen çarpışma nesnesini taşır."""
        msg = CollisionObject()

        if not isinstance(position, Point):
            position = Point(
                x=float(position[0]), y=float(position[1]), z=float(position[2])
            )
        if not isinstance(quat_xyzw, Quaternion):
            quat_xyzw = Quaternion(
                x=float(quat_xyzw[0]),
                y=float(quat_xyzw[1]),
                z=float(quat_xyzw[2]),
                w=float(quat_xyzw[3]),
            )

        pose = Pose()
        pose.position = position
        pose.orientation = quat_xyzw
        msg.pose = pose
        msg.id = id
        msg.operation = CollisionObject.MOVE
        msg.header.frame_id = (
            frame_id if frame_id is not None else self.__base_link_name
        )
        msg.header.stamp = self._node.get_clock().now().to_msg()

        self.__collision_object_publisher.publish(msg)

    def update_planning_scene(self) -> bool:
        """
        Güncel planlama sahnesini alır.

        Servis çağrısının başarılı olup olmadığını döndürür.
        """
        self._get_planning_scene_service.wait_for_service(timeout_sec=3.0)
        if not self._get_planning_scene_service.service_is_ready():
            self._node.get_logger().warn(
                f"Service '{self._get_planning_scene_service.srv_name}' is not yet available. "
                "Better luck next time!"
            )
            return False
        self.__planning_scene = self._get_planning_scene_service.call(
            GetPlanningScene.Request()
        ).scene
        return True

    def allow_collisions(self, id: str, allow: bool) -> Optional[Future]:
        """
        Planlama sahnesindeki bir öğenin çarpışmasına izin verir ya da engeller.

        İzin verilen çarpışma matrisini, o nesne ile diğer bütün nesneler arasındaki
        çarpışmalara izin verecek (ya da engelleyecek) biçimde değiştirir.

        `allow` True ise robot o nesneyle çarpışsa bile plan başarılı olur.
        `allow` False ise robot o nesneyle çarpışırsa plan başarısız olur.

        Servis çağrısının future'ını döndürür.
        """
        # Planlama sahnesini güncelle
        if not self.update_planning_scene():
            return None
        allowed_collision_matrix = self.__planning_scene.allowed_collision_matrix
        self.__old_allowed_collision_matrix = copy.deepcopy(allowed_collision_matrix)

        # Nesnenin izin verilen çarpışma matrisindeki yerini bul
        j = None
        if id not in allowed_collision_matrix.entry_names:
            allowed_collision_matrix.entry_names.append(id)
        else:
            j = allowed_collision_matrix.entry_names.index(id)
        # Diğer bütün nesneler için `id` nesnesiyle çarpışmaya izin ver (ya da engelle)
        for i in range(len(allowed_collision_matrix.entry_values)):
            if j is None:
                allowed_collision_matrix.entry_values[i].enabled.append(allow)
            elif i != j:
                allowed_collision_matrix.entry_values[i].enabled[j] = allow
        # `id` nesnesi için diğer bütün nesnelerle çarpışmaya izin ver (ya da engelle)
        allowed_collision_entry = AllowedCollisionEntry(
            enabled=[allow for _ in range(len(allowed_collision_matrix.entry_names))]
        )
        if j is None:
            allowed_collision_matrix.entry_values.append(allowed_collision_entry)
        else:
            allowed_collision_matrix.entry_values[j] = allowed_collision_entry

        # Yeni planlama sahnesini uygula
        self._apply_planning_scene_service.wait_for_service(timeout_sec=3.0)
        if not self._apply_planning_scene_service.service_is_ready():
            self._node.get_logger().warn(
                f"Service '{self._apply_planning_scene_service.srv_name}' is not yet available. "
                "Better luck next time!"
            )
            return None
        return self._apply_planning_scene_service.call_async(
            ApplyPlanningScene.Request(scene=self.__planning_scene)
        )

    def process_allow_collision_future(self, future: Future) -> bool:
        """
        Çarpışma izni servis çağrısının bitip başarılı olup olmadığını döndürür.

        Başarısız olduysa izin verilen çarpışma matrisi eski hâline döndürülür.
        """
        if not future.done():
            return False

        # Yanıtı al
        resp = future.result()

        # Başarısız olduysa eski planlama sahnesini geri yükle
        if not resp.success:
            self.__planning_scene.allowed_collision_matrix = (
                self.__old_allowed_collision_matrix
            )

        return resp.success

    def clear_all_collision_objects(self) -> Optional[Future]:
        """
        İliştirilmiş ve iliştirilmemiş bütün çarpışma nesnelerini planlama sahnesinden kaldırır.

        ApplyPlanningScene servis çağrısının future'ını döndürür.
        """
        # Planlama sahnesini güncelle
        if not self.update_planning_scene():
            return None
        self.__old_planning_scene = copy.deepcopy(self.__planning_scene)

        # Bütün çarpışma nesnelerini planlama sahnesinden kaldır
        self.__planning_scene.world.collision_objects = []
        self.__planning_scene.robot_state.attached_collision_objects = []

        # Yeni planlama sahnesini uygula
        self._apply_planning_scene_service.wait_for_service(timeout_sec=3.0)
        if not self._apply_planning_scene_service.service_is_ready():
            self._node.get_logger().warn(
                f"Service '{self._apply_planning_scene_service.srv_name}' is not yet available. "
                "Better luck next time!"
            )
            return None
        return self._apply_planning_scene_service.call_async(
            ApplyPlanningScene.Request(scene=self.__planning_scene)
        )

    def cancel_clear_all_collision_objects_future(self, future: Future) -> None:
        """Bütün çarpışma nesnelerini temizleme servis çağrısını iptal eder."""
        self._apply_planning_scene_service.remove_pending_request(future)

    def process_clear_all_collision_objects_future(self, future: Future) -> bool:
        """
        Bütün çarpışma nesnelerini temizleme çağrısının bitip başarılı olup olmadığını döndürür.

        Başarısız olduysa eski planlama sahnesi geri yüklenir.
        """
        if not future.done():
            return False

        # Yanıtı al
        resp = future.result()

        # Başarısız olduysa eski planlama sahnesini geri yükle
        if not resp.success:
            self.__planning_scene = self.__old_planning_scene

        return resp.success

    def __robot_description_callback(self, msg: String) -> None:
        if self.__urdf_xml is None:
            self.__urdf_xml = msg.data

    @property
    def joint_limits(self) -> dict:
        """
        /robot_description'dan ayrıştırılan {eklem_adı: JointLimit} sözlüğü.

        Model henüz gelmediyse (ya da ayrıştırılamadıysa) {} döner. Bir kez
        ayrıştırılır, sonra önbellekten verilir.
        """
        if self.__joint_limits is not None:
            return self.__joint_limits
        if self.__urdf_xml is None:
            return {}  # Henüz hata değil; kalıcı (latched) mesaj hâlâ yolda olabilir
        try:
            self.__joint_limits = parse_joint_limits(self.__urdf_xml)
        except Exception as e:
            self._node.get_logger().warn(
                f"Could not parse /robot_description ({e}); 2*pi joint-goal unwinding "
                "is DISABLED for this run (goals used exactly as given)."
            )
            self.__joint_limits = {}
            return self.__joint_limits
        wrappable = [
            n
            for n in self.__joint_names
            if n in self.__joint_limits and self.__joint_limits[n].wrappable
        ]
        self._node.get_logger().info(
            f"Joint limits parsed for {len(self.__joint_limits)} joints; "
            f"{len(wrappable)}/{len(self.__joint_names)} of this group's joints can be "
            f"unwound: {wrappable}"
        )
        return self.__joint_limits

    def __unwind_reference(
        self, reference: Optional[Union[JointState, List[float]]]
    ) -> Dict[str, float]:
        """
        "En yakın" ölçüsünün alınacağı {eklem_adı: açı} sözlüğü.

        `reference` bir JointState, bu grubun eklem sırasında düz bir liste ya da kolun
        güncel ölçülen durumu için None olabilir.
        """
        if reference is None:
            reference = self.joint_state
        if reference is None:
            return {}
        if isinstance(reference, JointState):
            return {
                n: float(p)
                for n, p in zip(reference.name, reference.position)
            }
        return {
            n: float(p) for n, p in zip(self.__joint_names, reference)
        }

    def unwind_joint_goal(
        self,
        joint_positions: List[float],
        joint_names: Optional[List[str]] = None,
        reference: Optional[Union[JointState, List[float]]] = None,
    ) -> List[float]:
        """
        Her dönel hedefi referans konfigürasyona EN YAKIN 2*pi eşdeğeriyle yeniden ifade eder.

        Eşdeğer, o eklemin URDF sınırları içinde kalmalıdır. theta ile theta +/- 2*pi
        AYNI fiziksel konfigürasyondur: bağlantı pozları, dolayısıyla çarpışmalar ve
        takım yönelimi aynıdır. Ama gidilecek sayılar çok farklıdır. IK rastgele bir dal
        döndürdüğü için hedef, kolun bulunduğu yerden gereksiz tam bir tur uzakta
        olabilir.

        Sınır denetimi bir ayrıntı değil, işin özüdür: bir eşdeğer YALNIZCA
        [lower + margin, upper - margin] içinde kalıyorsa kabul edilir. Böylece aralığı
        <= 2*pi olan eklemlerde (ikinci bir eşdeğere ulaşılamaz) fonksiyon kendiliğinden
        etkisiz kalır; doğrusal eklemlere ise hiç dokunulmaz, çünkü metreler sarılmaz.

        Tekrar uygulanabilir (idempotent): sarılmış bir hedefi aynı referansa göre
        yeniden sarmak bir şey kazandırmaz, dolayısıyla bir şey değiştirmez. Bu yüzden
        sarmayı zaten kendisi yapan çağıranlar (muayene yürütücüleri) etkilenmez.

        Sarma kapalıysa, URDF yoksa ya da yeniden yazmaya değer bir şey yoksa özgün
        liste değişmeden döner.
        """
        if not self.__unwind_joint_goals or joint_positions is None:
            return list(joint_positions) if joint_positions is not None else None
        limits = self.joint_limits
        if not limits:
            if not self.__unwind_warned:
                self.__unwind_warned = True
                self._node.get_logger().warn(
                    "No /robot_description yet -- joint limits unknown, so 2*pi "
                    "joint-goal unwinding is skipped (goals used exactly as given)."
                )
            return list(joint_positions)
        names = list(joint_names) if joint_names is not None else list(self.__joint_names)
        ref = self.__unwind_reference(reference)
        if not ref:
            return list(joint_positions)
        new_goal, changes = wrap_to_reference(
            list(joint_positions),
            names[: len(joint_positions)],
            ref,
            limits,
            margin=self.__unwind_limit_margin,
            min_gain=self.__unwind_min_gain,
        )
        if changes:
            self._node.get_logger().info(f"[unwind] {describe_changes(changes)}")
        return new_goal

    # ------------------------------------------------------------------ #
    # En yakın dal (nearest-branch) IK.
    # ------------------------------------------------------------------ #
    def _spin_or_sleep(self, timeout_sec: float = 1.0) -> None:
        """
        Engelleyici bir çağrı beklerken geri çağırmaların çalışmasına izin verir.

        Düğüm zaten çağıranın kendi yürütücüsündeyse (örneğin arka planda çalışan bir
        MultiThreadedExecutor) yalnızca uyunur ve geri çağırmaları o yürütücü dağıtır.
        Orada rclpy.spin_once() çağırmak aynı eylem istemcisini İKİ yürütücünün aynı
        anda beklemesine yol açar ve rclpy 'wait set index for status subscription is
        out of bounds' hatası verir; bu da çağıranın yürütücü iş parçacığını öldürür ve
        bütün abonelikleri (io_states, ...) dondurur.
        """
        executor = self._node.executor
        if (
            executor is not None
            and executor is not rclpy.get_global_executor()
            and self._node in executor.get_nodes()
        ):
            time.sleep(0.01)
        else:
            rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def __await_future(self, future: Optional[Future], timeout: float) -> bool:
        """
        Bir future'ı düğümü DÖNDÜRMEDEN (spin) bekler.

        Aşağıdaki her şey, düğümü zaten bir yürütücü altında olan bir çağırandan
        çalışır. Oradan rclpy.spin_once() çağırmak düğümü genel yürütücüye verir ve bir
        daha geri vermez; bu da çağıranın kendi aboneliklerini sessizce öldürür. Bu
        yüzden yoklama (polling) yapılır.
        """
        if future is None:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if future.done():
                return True
            time.sleep(0.01)
        return False

    def nearest_branch_ik(
        self,
        position: Union[Point, Tuple[float, float, float]],
        quat_xyzw: Union[Quaternion, Tuple[float, float, float, float]],
        ik_link_name: Optional[str] = None,
        planned_goal: Optional[List[float]] = None,
        random_seeds: int = 0,
        rng_seed: int = 0,
        seed_attempts: int = 1,
        ik_timeout: float = 0.1,
        avoid_collisions: bool = True,
        group_name: Optional[str] = None,
        lock_joints: Optional[dict] = None,
        unwind: bool = True,
        call_timeout: float = 2.0,
    ) -> List[Tuple[List[float], float]]:
        """
        TEK bir poza ulaşan eklem hedeflerini, gereken hareket miktarına göre sıralar.

        Sıralama, kolun şu anki yerinden ne kadar az hareketle varılacağına göredir.

        [(joint_positions, travel_rad), ...] döndürür, en yakını başta; pozun çözümü
        yoksa ya da eklem durumu bilinmiyorsa boş liste döner.

        Bir pozun birden çok geçerli IK dalı vardır (omuz / dirsek / bilek dönüşleri;
        fazladan eksenli bir kolda ise sürekli bir aile). Tek bir compute_ik çağrısı
        bunlardan BİRİNİ rastgele döndürür; birbirinden santimetreler uzaktaki iki bakış
        noktasının eklem uzayında yüzlerce derece uzak düşmesi bu yüzdendir. Bu fonksiyon
        dalları toplar ve sıralar.

        Farklı bir dalı iki şey üretir. Sorguyu tekrarlamak yalnızca çözücü RASTGELE
        YENİDEN BAŞLIYORSA bir şey örnekler (`mode: global` ile pick_ik başlar; KDL
        başlamaz, her seferinde aynı yanıtı verir); bu yüzden TOHUM da değiştirilir:
          - kolun ölçülen pozu,
          - çağıranın aklında bir çözüm varsa `planned_goal`,
          - URDF sınırlarından düzgün dağılımla çekilen `random_seeds` adet rastgele
            duruş; yerel bir çözücüyü tek yanıtından asıl kurtaran budur.
        `seed_attempts`, rastgele yeniden başlayan çözücüler için rastgele olmayan her
        tohumu tekrarlar.

        `lock_joints` ({ad: değer}) bu eklemleri her tohumda VE her sonuçta sabitler.
        Çözücünün oynatmaması gereken bir eksen için kullanın: rayın planlanan konumunu
        verirseniz dönen dalların hepsi o konumda kullanılabilir. `group_name` (ekseni
        içermeyen bir grup) ile birlikte kullanmak kaymayı yapısal olarak imkânsız kılar.

        Her çözüm puanlanmadan önce güncel poza doğru sarılır; böylece bir dal kazara
        oluşan tam tur farkıyla yargılanmaz. Sıralama, DÖNEL eklemler üzerinden
        joint_wrap.path_travel ile yapılır.

        Düğümü döndürmez (spin): bir yürütücü altında çalışan düğümden güvenle çağrılabilir.
        """
        current = self.joint_state
        if current is None:
            self._node.get_logger().warn(
                "nearest_branch_ik: no joint state yet; cannot rank branches."
            )
            return []
        names = list(self.__joint_names)
        limits = self.joint_limits
        reference = {n: float(p) for n, p in zip(current.name, current.position)}

        def as_state(values: Sequence[float]) -> JointState:
            js = JointState()
            js.name = list(names)
            js.position = [float(v) for v in values]
            return js

        def locked(values: Sequence[float]) -> List[float]:
            vals = [float(v) for v in values]
            if lock_joints:
                for i, n in enumerate(names):
                    if n in lock_joints:
                        vals[i] = float(lock_joints[n])
            return vals

        # --- tohumlar, en olasıdan başlayarak ---
        measured = [reference.get(n, 0.0) for n in names]
        seeds: List[Tuple[JointState, int]] = [
            (as_state(locked(measured)), max(1, int(seed_attempts)))
        ]
        if planned_goal is not None:
            seeds.append(
                (as_state(locked(planned_goal)), max(1, int(seed_attempts)))
            )
        if random_seeds > 0:
            base = list(planned_goal) if planned_goal is not None else list(measured)
            drawable = []
            for i, n in enumerate(names):
                if lock_joints and n in lock_joints:
                    continue
                limit = limits.get(n)
                if limit is not None and not limit.is_revolute:
                    # Doğrusal bir ekseni asla rastgele çekme: o seçilmiş bir değerdir
                    continue
                lo, hi = (
                    (float(limit.lower), float(limit.upper))
                    if limit is not None
                    else (-np.pi, np.pi)
                )
                if not (np.isfinite(lo) and np.isfinite(hi)) or hi - lo < 1e-6:
                    lo, hi = -np.pi, np.pi      # sürekli (continuous) eklem
                drawable.append((i, lo, hi))
            if drawable:
                rng = np.random.default_rng(int(rng_seed))
                for _ in range(int(random_seeds)):
                    vals = list(base)
                    for i, lo, hi in drawable:
                        vals[i] = float(rng.uniform(lo, hi))
                    seeds.append((as_state(locked(vals)), 1))
            elif limits:
                self._node.get_logger().warn(
                    "nearest_branch_ik: random_seeds requested but no revolute joint "
                    "has usable limits; random seeding is disabled for this call."
                )

        out: List[Tuple[List[float], float]] = []
        seen = set()
        for seed, repeats in seeds:
            for _ in range(repeats):
                future = self.compute_ik_async(
                    position=position,
                    quat_xyzw=quat_xyzw,
                    start_joint_state=seed,
                    ik_link_name=ik_link_name,
                    ik_timeout=ik_timeout,
                    avoid_collisions=avoid_collisions,
                    group_name=group_name,
                )
                if not self.__await_future(future, call_timeout):
                    continue
                # Sonucu get_compute_ik_result() üzerinden değil doğrudan oku: dal
                # aramasında rastgele tohumların çoğunun başarısız olması BEKLENİR ve o
                # yardımcı her başarısızlıkta uyarı basar, bu da günlüğü boğardı.
                response = future.result()
                if (
                    response is None
                    or response.error_code.val != MoveItErrorCodes.SUCCESS
                ):
                    continue
                solution = response.solution.joint_state
                lut = {n: float(p) for n, p in zip(solution.name, solution.position)}
                # Küçültülmüş bir grup yalnızca KENDİ eklemleriyle yanıt verir; geri kalanlar
                # çağıranın planladığı değeri korur (kilitli bir ray için tam doğrusu budur).
                # Çözücünün yanıt vermediği eklem, çağıranın planladığı değeri ya da
                # ölçülen değerini korur; iki durumda da hareket etmeyen bir eklemdir.
                fallback = list(planned_goal) if planned_goal is not None else measured
                goal = locked([
                    lut.get(n, float(fallback[i]) if i < len(fallback) else 0.0)
                    for i, n in enumerate(names)
                ])
                if unwind:
                    goal = self.unwind_joint_goal(
                        goal, joint_names=names, reference=current
                    )
                    goal = locked(goal)
                key = tuple(round(v, 3) for v in goal)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    (goal, path_travel(goal, names, reference, limits=limits))
                )
        out.sort(key=lambda t: t[1])
        return out

    def plan_nearest_branch(
        self,
        position: Union[Point, Tuple[float, float, float]],
        quat_xyzw: Union[Quaternion, Tuple[float, float, float, float]],
        max_candidates: int = 3,
        plan_timeout: float = 30.0,
        candidates: Optional[List[Tuple[List[float], float]]] = None,
        **branch_kwargs: Any,
    ) -> Optional[Tuple[JointTrajectory, List[float]]]:
        """
        Bir poza en yakın dalları üzerinden planlar ve gerçekten en az yol alan yörüngeyi tutar.

        (yörünge, eklem_hedefi) ya da None döndürür.

        Dalları hedef uzayındaki uzaklığa göre sıralamak yalnızca bir YAKLAŞIKLIKTIR:
        rastgele bir planlayıcının döndürdüğü yol o düz çizgiden çok uzaklaşabilir; bu
        yüzden en yakın hedef her zaman en kısa yolu vermez. Bu fonksiyon bunlardan en
        fazla `max_candidates` tanesini planlar ve yörüngelerin kendisini karşılaştırır.
        1 değeri "planlanabilen ilk dalı al" davranışına döner; her fazladan aday bir
        planlama çağrısı daha demektir.

        `candidates`, çağıranın nearest_branch_ik()'tan aldığı listeyi yeniden kullanmasını
        sağlar; verilmezse burada hesaplanır ve fazladan her anahtar kelime argümanı
        ona iletilir.

        Düğümü döndürmez (spin).
        """
        if candidates is None:
            candidates = self.nearest_branch_ik(
                position=position, quat_xyzw=quat_xyzw, **branch_kwargs
            )
        if not candidates:
            return None
        names = list(self.__joint_names)
        limits = self.joint_limits
        best = None
        for goal, _ in candidates[: max(1, int(max_candidates))]:
            future = self.plan_async(
                joint_positions=list(goal),
                joint_names=names,
                start_joint_state=self.joint_state,
            )
            if not self.__await_future(future, plan_timeout):
                continue
            trajectory = self.get_trajectory(future)
            if trajectory is None or not trajectory.points:
                continue
            travel = self.trajectory_travel(trajectory, limits=limits)
            if best is None or travel < best[0]:
                best = (travel, trajectory, list(goal))
        if best is None:
            return None
        return best[1], best[2]

    @staticmethod
    def trajectory_travel(
        trajectory: JointTrajectory, limits: Optional[dict] = None
    ) -> float:
        """
        Bir yörünge boyunca dönel eklemlerin toplam yolu [radyan].

        Daha kısa bir dalın küçültmeye çalıştığı maliyettir. `limits` doğrusal eksenleri
        tanıtıyorsa onlar hariç tutulur, çünkü bir metre ray ile bir radyan bilek
        karşılaştırılabilir büyüklükler değildir.
        """
        if trajectory is None or len(trajectory.points) < 2:
            return 0.0
        keep = [
            i
            for i, n in enumerate(trajectory.joint_names)
            if not (limits and n in limits and not limits[n].is_revolute)
        ]
        if not keep:
            return 0.0
        positions = np.array([p.positions for p in trajectory.points], dtype=float)
        return float(np.sum(np.abs(np.diff(positions[:, keep], axis=0))))

    def __joint_state_callback(self, msg: JointState) -> None:
        # Yalnızca ilgili bütün eklemler mesajda varsa güncelle
        for joint_name in self.joint_names:
            if joint_name not in msg.name:
                return

        self.__joint_state_mutex.acquire()
        self.__joint_state = msg
        self.__new_joint_state_available = True
        self.__joint_state_mutex.release()

    def _plan_kinematic_path(self) -> Optional[Future]:
        # Hareket eylemi hedefindeki isteği yeniden kullan
        self.__kinematic_path_request.motion_plan_request = (
            self.__move_action_goal.request
        )

        stamp = self._node.get_clock().now().to_msg()
        self.__kinematic_path_request.motion_plan_request.workspace_parameters.header.stamp = (
            stamp
        )
        for (
            constraints
        ) in self.__kinematic_path_request.motion_plan_request.goal_constraints:
            for position_constraint in constraints.position_constraints:
                position_constraint.header.stamp = stamp
            for orientation_constraint in constraints.orientation_constraints:
                orientation_constraint.header.stamp = stamp
        self._plan_kinematic_path_service.wait_for_service(timeout_sec=3.0)
        if not self._plan_kinematic_path_service.service_is_ready():
            self._node.get_logger().warn(
                f"Service '{self._plan_kinematic_path_service.srv_name}' is not yet available. "
                "Better luck next time!"
            )
            return None

        return self._plan_kinematic_path_service.call_async(
            self.__kinematic_path_request
        )

    def _plan_cartesian_path(
        self,
        max_step: float = 0.0025,
        frame_id: Optional[str] = None,
    ) -> Optional[Future]:
        # Hareket eylemi hedefindeki isteği yeniden kullan
        self.__cartesian_path_request.start_state = (
            self.__move_action_goal.request.start_state
        )

        # Aşağıdaki öznitelikler Iron'da eklendi; Humble'da yoklar.
        if hasattr(self.__cartesian_path_request, "max_velocity_scaling_factor"):
            self.__cartesian_path_request.max_velocity_scaling_factor = (
                self.__move_action_goal.request.max_velocity_scaling_factor
            )
        if hasattr(self.__cartesian_path_request, "max_acceleration_scaling_factor"):
            self.__cartesian_path_request.max_acceleration_scaling_factor = (
                self.__move_action_goal.request.max_acceleration_scaling_factor
            )

        self.__cartesian_path_request.group_name = (
            self.__move_action_goal.request.group_name
        )
        self.__cartesian_path_request.link_name = self.__end_effector_name
        self.__cartesian_path_request.max_step = max_step

        self.__cartesian_path_request.header.frame_id = (
            frame_id if frame_id is not None else self.__base_link_name
        )

        stamp = self._node.get_clock().now().to_msg()
        self.__cartesian_path_request.header.stamp = stamp

        self.__cartesian_path_request.path_constraints = (
            self.__move_action_goal.request.path_constraints
        )
        for (
            position_constraint
        ) in self.__cartesian_path_request.path_constraints.position_constraints:
            position_constraint.header.stamp = stamp
        for (
            orientation_constraint
        ) in self.__cartesian_path_request.path_constraints.orientation_constraints:
            orientation_constraint.header.stamp = stamp
        # joint_constraint mesaj tipinde başlık (header) yok

        target_pose = Pose()
        target_pose.position = (
            self.__move_action_goal.request.goal_constraints[-1]
            .position_constraints[-1]
            .constraint_region.primitive_poses[0]
            .position
        )
        target_pose.orientation = (
            self.__move_action_goal.request.goal_constraints[-1]
            .orientation_constraints[-1]
            .orientation
        )

        self.__cartesian_path_request.waypoints = [target_pose]
        self._plan_cartesian_path_service.wait_for_service(timeout_sec=3.0)
        if not self._plan_cartesian_path_service.service_is_ready():
            self._node.get_logger().warn(
                f"Service '{self._plan_cartesian_path_service.srv_name}' is not yet available. "
                "Better luck next time!"
            )
            return None

        return self._plan_cartesian_path_service.call_async(
            self.__cartesian_path_request
        )

    def _send_goal_async_move_action(self) -> None:
        self.__execution_mutex.acquire()
        stamp = self._node.get_clock().now().to_msg()
        self.__move_action_goal.request.workspace_parameters.header.stamp = stamp
        if not self.__move_action_client.server_is_ready():
            self._node.get_logger().warn(
                f"Action server '{self.__move_action_client._action_name}' is not yet available. "
                "Better luck next time!"
            )
            return

        self.__last_error_code = None
        self.__is_motion_requested = True
        self.__send_goal_future_move_action = self.__move_action_client.send_goal_async(
            goal=self.__move_action_goal,
            feedback_callback=None,
        )

        self.__send_goal_future_move_action.add_done_callback(
            self.__response_callback_move_action
        )

        self.__execution_mutex.release()

    def __response_callback_move_action(self, response: Future) -> None:
        self.__execution_mutex.acquire()
        goal_handle = response.result()
        if not goal_handle.accepted:
            self._node.get_logger().warn(
                f"Action '{self.__move_action_client._action_name}' was rejected."
            )
            self.__is_motion_requested = False
            return

        self.__execution_goal_handle = goal_handle
        self.__is_executing = True
        self.__is_motion_requested = False

        self.__get_result_future_move_action = goal_handle.get_result_async()
        self.__get_result_future_move_action.add_done_callback(
            self.__result_callback_move_action
        )
        self.__execution_mutex.release()

    def __result_callback_move_action(self, res: Future) -> None:
        self.__execution_mutex.acquire()
        if res.result().status != GoalStatus.STATUS_SUCCEEDED:
            self._node.get_logger().warn(
                f"Action '{self.__move_action_client._action_name}' was unsuccessful: "
                f"{enum_to_str(GoalStatus, res.result().status)}."
            )
            self.motion_suceeded = False
        else:
            self.motion_suceeded = True

        self.__last_error_code = res.result().result.error_code

        self.__execution_goal_handle = None
        self.__is_executing = False
        self.__execution_mutex.release()

    def _send_goal_async_execute_trajectory(
        self,
        goal: ExecuteTrajectory,
    ) -> None:
        self.__execution_mutex.acquire()

        if not self._execute_trajectory_action_client.server_is_ready():
            self._node.get_logger().warn(
                f"Action server '{self._execute_trajectory_action_client._action_name}' is not "
                "yet available. Better luck next time!"
            )
            return

        self.__last_error_code = None
        self.__is_motion_requested = True
        self.__send_goal_future_execute_trajectory = (
            self._execute_trajectory_action_client.send_goal_async(
                goal=goal,
                feedback_callback=None,
            )
        )

        self.__send_goal_future_execute_trajectory.add_done_callback(
            self.__response_callback_execute_trajectory
        )
        self.__execution_mutex.release()

    def __response_callback_execute_trajectory(self, response: Future) -> None:
        self.__execution_mutex.acquire()
        goal_handle = response.result()
        if not goal_handle.accepted:
            self._node.get_logger().warn(
                f"Action '{self._execute_trajectory_action_client._action_name}' was rejected."
            )
            self.__is_motion_requested = False
            return

        self.__execution_goal_handle = goal_handle
        self.__is_executing = True
        self.__is_motion_requested = False

        self.__get_result_future_execute_trajectory = goal_handle.get_result_async()
        self.__get_result_future_execute_trajectory.add_done_callback(
            self.__result_callback_execute_trajectory
        )
        self.__execution_mutex.release()

    def __result_callback_execute_trajectory(self, res: Future) -> None:
        self.__execution_mutex.acquire()
        if res.result().status != GoalStatus.STATUS_SUCCEEDED:
            self._node.get_logger().warn(
                f"Action '{self._execute_trajectory_action_client._action_name}' was "
                f"unsuccessful: {enum_to_str(GoalStatus, res.result().status)}."
            )
            self.motion_suceeded = False
        else:
            self.motion_suceeded = True

        self.__last_error_code = res.result().result.error_code

        self.__execution_goal_handle = None
        self.__is_executing = False
        self.__execution_mutex.release()

    @classmethod
    def __init_move_action_goal(
        cls, frame_id: str, group_name: str, end_effector: str
    ) -> MoveGroup.Goal:
        move_action_goal = MoveGroup.Goal()
        move_action_goal.request.workspace_parameters.header.frame_id = frame_id
        # move_action_goal.request.workspace_parameters.header.stamp = istek sırasında atanır
        move_action_goal.request.workspace_parameters.min_corner.x = -1.0
        move_action_goal.request.workspace_parameters.min_corner.y = -1.0
        move_action_goal.request.workspace_parameters.min_corner.z = -1.0
        move_action_goal.request.workspace_parameters.max_corner.x = 1.0
        move_action_goal.request.workspace_parameters.max_corner.y = 1.0
        move_action_goal.request.workspace_parameters.max_corner.z = 1.0
        # move_action_goal.request.start_state = istek sırasında atanır
        move_action_goal.request.goal_constraints = [Constraints()]
        move_action_goal.request.path_constraints = Constraints()
        # move_action_goal.request.trajectory_constraints = kullanılmıyor
        # move_action_goal.request.reference_trajectories = kullanılmıyor
        move_action_goal.request.pipeline_id = ""
        move_action_goal.request.planner_id = ""
        move_action_goal.request.group_name = group_name
        move_action_goal.request.num_planning_attempts = 5
        move_action_goal.request.allowed_planning_time = 0.5
        move_action_goal.request.max_velocity_scaling_factor = 0.0
        move_action_goal.request.max_acceleration_scaling_factor = 0.0
        # Not: Öznitelik Iron'da yeniden adlandırıldı
        # (https://github.com/ros-planning/moveit_msgs/pull/130)
        if hasattr(move_action_goal.request, "cartesian_speed_limited_link"):
            move_action_goal.request.cartesian_speed_limited_link = end_effector
        else:
            move_action_goal.request.cartesian_speed_end_effector_link = end_effector
        move_action_goal.request.max_cartesian_speed = 0.0

        # move_action_goal.planning_options.planning_scene_diff = kullanılmıyor
        move_action_goal.planning_options.plan_only = False
        # move_action_goal.planning_options.look_around = kullanılmıyor
        # move_action_goal.planning_options.look_around_attempts = kullanılmıyor
        # move_action_goal.planning_options.max_safe_execution_cost = kullanılmıyor
        # move_action_goal.planning_options.replan = kullanılmıyor
        # move_action_goal.planning_options.replan_attempts = kullanılmıyor
        # move_action_goal.planning_options.replan_delay = kullanılmıyor

        return move_action_goal

    def __init_compute_fk(self) -> None:
        self.__compute_fk_client = self._node.create_client(
            srv_type=GetPositionFK,
            srv_name="compute_fk",
            callback_group=self._callback_group,
        )

        self.__compute_fk_req = GetPositionFK.Request()
        self.__compute_fk_req.header.frame_id = self.__base_link_name
        # self.__compute_fk_req.header.stamp = istek sırasında atanır
        # self.__compute_fk_req.fk_link_names = istek sırasında atanır
        # self.__compute_fk_req.robot_state.joint_state = istek sırasında atanır
        # self.__compute_fk_req.robot_state.multi_dof_ = kullanılmıyor
        # self.__compute_fk_req.robot_state.attached_collision_objects = kullanılmıyor
        self.__compute_fk_req.robot_state.is_diff = False

    def __init_compute_ik(self) -> None:
        # IK için servis istemcisi
        self.__compute_ik_client = self._node.create_client(
            srv_type=GetPositionIK,
            srv_name="compute_ik",
            callback_group=self._callback_group,
        )

        self.__compute_ik_req = GetPositionIK.Request()
        self.__compute_ik_req.ik_request.group_name = self.__group_name
        # self.__compute_ik_req.ik_request.robot_state.joint_state = istek sırasında atanır
        # self.__compute_ik_req.ik_request.robot_state.multi_dof_ = kullanılmıyor
        # self.__compute_ik_req.ik_request.robot_state.attached_collision_objects = kullanılmıyor
        self.__compute_ik_req.ik_request.robot_state.is_diff = False
        # self.__compute_ik_req.ik_request.constraints = istek sırasında atanır YA DA kullanılmıyor
        self.__compute_ik_req.ik_request.avoid_collisions = True
        # self.__compute_ik_req.ik_request.ik_link_name = kullanılmıyor
        self.__compute_ik_req.ik_request.pose_stamped.header.frame_id = (
            self.__base_link_name
        )
        # self.__compute_ik_req.ik_request.pose_stamped.header.stamp = istek sırasında atanır
        # self.__compute_ik_req.ik_request.pose_stamped.pose = istek sırasında atanır
        # self.__compute_ik_req.ik_request.ik_link_names = kullanılmıyor
        # self.__compute_ik_req.ik_request.pose_stamped_vector = kullanılmıyor
        # self.__compute_ik_req.ik_request.timeout.sec = kullanılmıyor
        # self.__compute_ik_req.ik_request.timeout.nanosec = kullanılmıyor

    @property
    def planning_scene(self) -> Optional[PlanningScene]:
        return self.__planning_scene

    @property
    def follow_joint_trajectory_action_client(self) -> str:
        return self.__follow_joint_trajectory_action_client

    @property
    def end_effector_name(self) -> str:
        return self.__end_effector_name

    @property
    def base_link_name(self) -> str:
        return self.__base_link_name

    @property
    def joint_names(self) -> List[str]:
        return self.__joint_names

    @property
    def joint_state(self) -> Optional[JointState]:
        self.__joint_state_mutex.acquire()
        joint_state = self.__joint_state
        self.__joint_state_mutex.release()
        return joint_state

    @property
    def new_joint_state_available(self) -> bool:
        return self.__new_joint_state_available

    @property
    def max_velocity(self) -> float:
        return self.__move_action_goal.request.max_velocity_scaling_factor

    @max_velocity.setter
    def max_velocity(self, value: float) -> None:
        self.__move_action_goal.request.max_velocity_scaling_factor = value

    @property
    def max_acceleration(self) -> float:
        return self.__move_action_goal.request.max_acceleration_scaling_factor

    @max_acceleration.setter
    def max_acceleration(self, value: float) -> None:
        self.__move_action_goal.request.max_acceleration_scaling_factor = value

    @property
    def num_planning_attempts(self) -> int:
        return self.__move_action_goal.request.num_planning_attempts

    @num_planning_attempts.setter
    def num_planning_attempts(self, value: int) -> None:
        self.__move_action_goal.request.num_planning_attempts = value

    @property
    def allowed_planning_time(self) -> float:
        return self.__move_action_goal.request.allowed_planning_time

    @allowed_planning_time.setter
    def allowed_planning_time(self, value: float) -> None:
        self.__move_action_goal.request.allowed_planning_time = value

    @property
    def cartesian_avoid_collisions(self) -> bool:
        return self.__cartesian_path_request.request.avoid_collisions

    @cartesian_avoid_collisions.setter
    def cartesian_avoid_collisions(self, value: bool) -> None:
        self.__cartesian_path_request.avoid_collisions = value

    @property
    def cartesian_jump_threshold(self) -> float:
        return self.__cartesian_path_request.request.jump_threshold

    @cartesian_jump_threshold.setter
    def cartesian_jump_threshold(self, value: float) -> None:
        self.__cartesian_path_request.jump_threshold = value

    @property
    def cartesian_prismatic_jump_threshold(self) -> float:
        return self.__cartesian_path_request.request.prismatic_jump_threshold

    @cartesian_prismatic_jump_threshold.setter
    def cartesian_prismatic_jump_threshold(self, value: float) -> None:
        self.__cartesian_path_request.prismatic_jump_threshold = value

    @property
    def cartesian_revolute_jump_threshold(self) -> float:
        return self.__cartesian_path_request.request.revolute_jump_threshold

    @cartesian_revolute_jump_threshold.setter
    def cartesian_revolute_jump_threshold(self, value: float) -> None:
        self.__cartesian_path_request.revolute_jump_threshold = value

    @property
    def pipeline_id(self) -> int:
        return self.__move_action_goal.request.pipeline_id

    @pipeline_id.setter
    def pipeline_id(self, value: str) -> None:
        self.__move_action_goal.request.pipeline_id = value

    @property
    def planner_id(self) -> int:
        return self.__move_action_goal.request.planner_id

    @planner_id.setter
    def planner_id(self, value: str) -> None:
        self.__move_action_goal.request.planner_id = value


def init_joint_state(
    joint_names: List[str],
    joint_positions: Optional[List[str]] = None,
    joint_velocities: Optional[List[str]] = None,
    joint_effort: Optional[List[str]] = None,
) -> JointState:
    joint_state = JointState()

    joint_state.name = joint_names
    joint_state.position = (
        joint_positions if joint_positions is not None else [0.0] * len(joint_names)
    )
    joint_state.velocity = (
        joint_velocities if joint_velocities is not None else [0.0] * len(joint_names)
    )
    joint_state.effort = (
        joint_effort if joint_effort is not None else [0.0] * len(joint_names)
    )

    return joint_state


def init_execute_trajectory_goal(
    joint_trajectory: JointTrajectory,
) -> Optional[ExecuteTrajectory.Goal]:
    if joint_trajectory is None:
        return None

    execute_trajectory_goal = ExecuteTrajectory.Goal()

    execute_trajectory_goal.trajectory.joint_trajectory = joint_trajectory

    return execute_trajectory_goal


def init_dummy_joint_trajectory_from_state(
    joint_state: JointState, duration_sec: int = 0, duration_nanosec: int = 0
) -> JointTrajectory:
    joint_trajectory = JointTrajectory()
    joint_trajectory.joint_names = joint_state.name

    point = JointTrajectoryPoint()
    point.positions = joint_state.position
    point.velocities = joint_state.velocity
    point.accelerations = [0.0] * len(joint_trajectory.joint_names)
    point.effort = joint_state.effort
    point.time_from_start.sec = duration_sec
    point.time_from_start.nanosec = duration_nanosec
    joint_trajectory.points.append(point)

    return joint_trajectory
