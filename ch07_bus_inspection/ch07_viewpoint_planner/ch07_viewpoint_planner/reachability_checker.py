from typing import Any, Dict, List, Optional, Sequence, Tuple

from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState

# /compute_ik'nın en sık döndürdüğü MoveItErrorCodes değerlerinin okunabilir
# adları; böylece hata kayıtları IK'nın *neden* başarısız olduğunu (çözüm yok /
# çarpışma / zaman aşımı) ham bir tamsayı yerine açıklar.
IK_ERROR_NAMES = {
    1: 'SUCCESS',
    -31: 'NO_IK_SOLUTION',
    -12: 'GOAL_IN_COLLISION',
    -13: 'GOAL_VIOLATES_PATH_CONSTRAINTS',
    -14: 'GOAL_CONSTRAINTS_VIOLATED',
    -10: 'START_STATE_IN_COLLISION',
    -9: 'TIMED_OUT',
    -17: 'FRAME_TRANSFORM_FAILURE',
    -18: 'COLLISION_CHECKING_UNAVAILABLE',
    -19: 'ROBOT_STATE_STALE',
    -21: 'INVALID_GROUP_NAME',
    -22: 'INVALID_GOAL_CONSTRAINTS',
    -23: 'INVALID_ROBOT_STATE',
    -24: 'INVALID_LINK_NAME',
}


class ReachabilityChecker(Node):
    """Bir bakış noktasının robotça erişilebilir olup olmadığını denetler."""

    def __init__(self, group_name: str = "real_ur10e", base_frame: str = "world",
                 tool_frame: str = "ur10e_depth_optical_frame",
                 avoid_collisions: bool = True) -> None:
        super().__init__(f'reachability_checker_{group_name}')
        self.group_name = group_name
        self.base_frame = base_frame
        self.tool_frame = tool_frame
        # True iken IK yalnızca kolun tamamı o pozda çarpışmasızsa başarılı olur.
        # Kafes benzeri bir şasenin yakın mesafeden muayenesinde bu, kamera pozunun
        # kendisi kinematik olarak erişilebilir olsa bile KOL gövdesinin yapının
        # içine gireceği pek çok pozu reddeder. Saf kinematik erişilebilirliği ölçmek
        # için False verin (tanı amaçlı ya da çarpışma geometrisine henüz
        # güvenilemediğinde).
        self.avoid_collisions = avoid_collisions

        # Bütün check_ik çağrıları boyunca {error_code: count} biriktirir; böylece
        # çağıranlar IK'nın *neden* başarısız olduğunun dökümünü yazabilir
# (bkz. log_error_breakdown).
        self.ik_error_counts = {}

        self.get_logger().info(
            f"Initializing ReachabilityChecker: group='{group_name}', base_frame='{base_frame}', "
            f"tool_frame(ik_link)='{tool_frame}'."
        )
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')

    def wait_for_service(self, timeout_sec: float = 5.0) -> bool:
        available = self.ik_client.wait_for_service(timeout_sec=timeout_sec)
        if not available:
            self.get_logger().error(
                f"/compute_ik service did not become available within {timeout_sec}s "
                f"(group='{self.group_name}'). Is move_group running?"
            )
        return available

    def check_ik(self, position: Sequence[float], orientation: Sequence[float],
                 timeout: float = 0.1, seed_joint_state: Optional[JointState] = None,
                 count_errors: bool = True) -> Tuple[bool, Optional[JointState]]:
        """
        Verilen pozun (konum, yönelim) erişilebilir olup olmadığını denetler.

        orientation bir kuaterniyon [x, y, z, w] olmalıdır.

        `seed_joint_state` (bir JointState), çözücünün aramaya başladığı
        konfigürasyondur. Bir pozun, ona bambaşka kol konfigürasyonlarından ulaşan
        birkaç geçerli IK dalı vardır (omuz/dirsek/bilek dönüşleri) ve tohum olmadan
        çözücü rastgele birini döndürür; birbirinden santimetreler uzaktaki iki bakış
        noktasının eklem uzayında yüzlerce derece uzak düşmesi bu yüzdendir. Bir önceki
        bakış noktasının çözümünü tohum olarak vermek, ona yakın bir dal ister.

        `count_errors` False iken başarısız bir deneme ik_error_counts dökümüne
        eklenmez; böylece süreklilik için yapılan yeniden çözümler erişilebilirlik
        istatistiklerini kirletmez.
        """
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.timeout.sec = int(timeout)
        req.ik_request.timeout.nanosec = int((timeout - int(timeout)) * 1e9)
        req.ik_request.avoid_collisions = self.avoid_collisions
        if seed_joint_state is not None:
            req.ik_request.robot_state.joint_state = seed_joint_state
        # Grubun varsayılan uç bağlantısı yerine asıl algılayıcı/takım çerçevesi için
        # çöz; bu olmadan istek tool_frame'i sessizce yok sayar ve çok uçlu
        # gruplarda IK'yı yanlış bağlantı için çözebilir.
        if self.tool_frame:
            req.ik_request.ik_link_name = self.tool_frame

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.base_frame
        pose_stamped.pose.position.x = float(position[0])
        pose_stamped.pose.position.y = float(position[1])
        pose_stamped.pose.position.z = float(position[2])

        pose_stamped.pose.orientation.x = float(orientation[0])
        pose_stamped.pose.orientation.y = float(orientation[1])
        pose_stamped.pose.orientation.z = float(orientation[2])
        pose_stamped.pose.orientation.w = float(orientation[3])

        req.ik_request.pose_stamped = pose_stamped

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        if response is None:
            self.get_logger().error(
                "/compute_ik call returned no response (timed out or service crashed) for group "
                f"'{self.group_name}'."
            )
            return False, None

        if response.error_code.val == response.error_code.SUCCESS:
            self.get_logger().debug(
                f"IK success for group '{self.group_name}' at position={list(position)}."
            )
            return True, response.solution.joint_state
        else:
            err = response.error_code.val
            if count_errors:
                self.ik_error_counts[err] = self.ik_error_counts.get(err, 0) + 1
            self.get_logger().debug(
                f"IK failed for group '{self.group_name}' at position={list(position)}: "
                f"error_code={err} ({IK_ERROR_NAMES.get(err, 'UNKNOWN')})."
            )
            return False, None

    def log_error_breakdown(self) -> None:
        """
        Şimdiye kadarki her IK hatasının hata koduna göre özetini yazar.

        Düşük kaplamayı teşhis etmeye yarar: örneğin çoğunlukla NO_IK_SOLUTION, kolun
        pozlara fiziksel olarak ulaşamadığını; GOAL_IN_COLLISION ise pozların
        erişilebilir olduğunu ama sahne/öz çarpışma tarafından engellendiğini gösterir.
        """
        if not self.ik_error_counts:
            return
        breakdown = ', '.join(
            f"{IK_ERROR_NAMES.get(code, 'UNKNOWN')}({code})={count}"
            for code, count in sorted(self.ik_error_counts.items(), key=lambda kv: -kv[1])
        )
        self.get_logger().info(
            f"IK failure breakdown for group '{self.group_name}': {breakdown}."
        )
        # Başarısızlıkların *tamamı* NO_IK_SOLUTION ise bu çoğu zaman gerçek bir
        # erişilemezlik değil, bir yapılandırma sorunudur; kullanıcı boşuna bakış
        # noktası ayarıyla uğraşmasın diye olağan şüphelileri göster.
        if set(self.ik_error_counts) == {-31}:
            self.get_logger().warning(
                f"All IK failures for group '{self.group_name}' are NO_IK_SOLUTION(-31). "
                "If NONE succeed, this is usually a config issue, not the viewpoints: check "
                f"(1) a kinematics_solver is defined for '{self.group_name}' in kinematics.yaml "
                "(move_group logs 'No kinematics solver instantiated' otherwise), "
                f"(2) the group's base link is connected to base_frame '{self.base_frame}' in the "
                "TF tree, and (3) kinematics_solver_timeout is not too small (KDL often needs "
                ">=0.05s)."
            )

    def filter_reachable_viewpoints(self, viewpoints: List[Dict[str, Any]]
                                    ) -> List[Dict[str, Any]]:
        """Erişilemeyen bakış noktalarını eler ve eklem çözümlerini ekler."""
        reachable_viewpoints = []
        unreachable_count = 0

        if not self.wait_for_service():
            # Bu fonksiyon eskiden IK'yı hiç denetlemeden bütün girdi bakış noktalarını
            # sessizce "erişilebilir" olarak döndürüyordu; bu, gerçek erişilemezlik
            # sorunlarını gizliyordu. Bunun yerine yüksek sesle hata ver ki çağıranlar
            # ne yapacaklarına (örneğin planlamayı durdurmaya) karar verebilsin ve
            # doğrulanmamış pozlarla bir plan üretilmesin.
            self.get_logger().error(
                f"/compute_ik service not available for group '{self.group_name}'. "
                f"Refusing to mark {len(viewpoints)} viewpoints as reachable without verification."
            )
            raise RuntimeError(
                f"IK service unavailable for group '{self.group_name}'; cannot verify "
                "reachability."
            )

        self.get_logger().info(
            f"Checking IK reachability for {len(viewpoints)} viewpoints "
            f"(group='{self.group_name}')..."
        )
        for i, vp in enumerate(viewpoints):
            pos = vp['position']
            rot_mat = vp['rotation']

            r = Rotation.from_matrix(rot_mat)
            quat = r.as_quat()  # [x, y, z, w]

            success, joint_state = self.check_ik(pos, quat)

            if success:
                vp['joint_solution'] = joint_state
                reachable_viewpoints.append(vp)
            else:
                unreachable_count += 1

            if (i + 1) % 10 == 0:
                self.get_logger().info(f"Checked IK for {i+1} viewpoints...")

        total = len(viewpoints)
        unreachable_pct = 100 * unreachable_count / total if total else 0.0
        self.get_logger().info(
            f"IK filtering complete for group '{self.group_name}'. "
            f"{len(reachable_viewpoints)} reachable, {unreachable_count} unreachable "
            f"({unreachable_pct:.1f}%)."
        )
        self.log_error_breakdown()
        if unreachable_pct > 25.0:
            self.get_logger().warning(
                f"{unreachable_pct:.1f}% of candidate viewpoints were unreachable for group "
                f"'{self.group_name}'. This will reduce achievable coverage -- consider "
                "revisiting "
                "viewpoint_distances/tilt_variations or the robot's mounting position."
            )
        return reachable_viewpoints
