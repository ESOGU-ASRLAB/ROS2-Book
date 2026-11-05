"""
Kolu birkaç ara noktadan geçiren örnek yörünge istemcisi.

joint_trajectory_controller'ın FollowJointTrajectory eylemine bir yörünge
gönderir. Aynı düğüm, hiçbir değişiklik yapmadan hem mock donanımda hem
Gazebo'da çalışır; ros2_control'ün donanım soyutlamasının gösterdiği de
budur (Bölüm 5.4).

Çalıştırma:
    # Mock kipinde
    $ ros2 run ch05_simulation_visualization send_trajectory
    # Gazebo kipinde (zaman Gazebo'dan gelir)
    $ ros2 run ch05_simulation_visualization send_trajectory \
        --ros-args -p use_sim_time:=true
"""

from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint

JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3',
               'joint_4', 'joint_5', 'joint_6']

# (saniye, eklem açıları [rad]) - başlangıçtan itibaren varış zamanları
WAYPOINTS: list[tuple[float, list[float]]] = [
    (3.0, [0.8, 0.4, 0.6, 0.0, 0.5, 0.0]),
    (6.0, [-0.8, 0.3, 0.9, 0.5, -0.4, 1.0]),
    (9.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
]


class TrajectorySender(Node):
    """FollowJointTrajectory eylemiyle örnek bir yörünge yürütür."""

    def __init__(self) -> None:
        super().__init__('trajectory_sender')
        self._client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory')

    def run(self) -> bool:
        """Yörüngeyi gönderir; başarıyla biterse True döndürür."""
        if not self._client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                'Yörünge denetleyicisi yok: cell_bringup çalışıyor mu?')
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        for t_sec, positions in WAYPOINTS:
            point = JointTrajectoryPoint()
            point.positions = positions
            point.time_from_start = DurationMsg(
                sec=int(t_sec), nanosec=int((t_sec % 1.0) * 1e9))
            goal.trajectory.points.append(point)

        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Yörünge reddedildi')
            return False
        self.get_logger().info(
            f'{len(WAYPOINTS)} noktalı yörünge yürütülüyor...')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=WAYPOINTS[-1][0] + 10.0)
        if result_future.result() is None:
            self.get_logger().error('Sonuç zaman aşımına uğradı')
            return False
        # error_code 0 = SUCCESSFUL; diğer kodlar FollowJointTrajectory
        # arayüzünde tanımlıdır (ör. -4 PATH_TOLERANCE_VIOLATED).
        error_code = result_future.result().result.error_code
        self.get_logger().info(f'Bitti, error_code={error_code}')
        return error_code == FollowJointTrajectory.Result.SUCCESSFUL


def main(args: list[str] | None = None) -> None:
    """Yörüngeyi bir kez gönderip çıkar."""
    rclpy.init(args=args)
    node = TrajectorySender()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
