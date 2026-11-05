"""
Liste 9 ve Liste 10 - Eylem istemcisi, sonuç durumları ve güvenli iptal.

turtlesim'in /turtle1/rotate_absolute eylemine bir hedef açı gönderir,
geri beslemeyi (kalan açı) günlüğe yazar ve sonucu bekler.
cancel_after_sec > 0 verilirse hedef o süre sonunda iptal edilir ve
kaplumbağa gerçekten duruncaya kadar beklenir (Liste 10).

Çalıştırma (1. terminalde turtlesim açıkken):
    $ ros2 run turtlesim turtlesim_node
    $ ros2 run ch04_ros2_communication rotate_client
    $ ros2 run ch04_ros2_communication rotate_client --ros-args \
        -p target_theta:=-3.0 -p cancel_after_sec:=0.5
"""

import time

from action_msgs.msg import GoalStatus
import rclpy
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from turtlesim.action import RotateAbsolute
from turtlesim.msg import Pose

# Sonuç durum kodlarının okunabilir adları (Tablo 2)
STATUS_NAMES = {
    GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
    GoalStatus.STATUS_ABORTED: 'ABORTED',
    GoalStatus.STATUS_CANCELED: 'CANCELED',
}


class RotateClient(Node):
    """RotateAbsolute eylemine hedef gönderen ve gerekirse iptal eden düğüm."""

    def __init__(self) -> None:
        super().__init__('rotate_client')
        self.declare_parameter('target_theta', 1.57)      # [rad]
        self.declare_parameter('cancel_after_sec', 0.0)   # 0: iptal etme
        self._client = ActionClient(
            self, RotateAbsolute, '/turtle1/rotate_absolute')
        self._goal_handle: ClientGoalHandle | None = None
        # Güvenli durdurma için kaplumbağanın anlık açısal hızı izlenir;
        # gerçek bir kolda bu, eklem hızlarının en büyüğü olurdu.
        self._angular_speed = 0.0
        self.create_subscription(Pose, '/turtle1/pose', self._on_pose, 10)

    def _on_pose(self, msg: Pose) -> None:
        self._angular_speed = abs(msg.angular_velocity)

    def _on_feedback(self, msg: RotateAbsolute.Impl.FeedbackMessage) -> None:
        self.get_logger().info(f'kalan: {msg.feedback.remaining:.2f} rad')

    def rotate(self) -> bool:
        """Hedefi gönderir; iş başarıyla biterse True döndürür."""
        # 1) Sunucu ayakta mı?
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Eylem sunucusu yok')
            return False

        # 2) Hedefi gönder ve kabul edilip edilmediğini öğren
        goal = RotateAbsolute.Goal()
        goal.theta = float(self.get_parameter('target_theta').value)
        send_future = self._client.send_goal_async(
            goal, feedback_callback=self._on_feedback)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
        self._goal_handle = send_future.result()
        if self._goal_handle is None or not self._goal_handle.accepted:
            self.get_logger().warn('Hedef reddedildi')
            return False

        # İptal isteniyorsa hedefi bir süre çalıştırıp durdur (Liste 10)
        cancel_after = float(self.get_parameter('cancel_after_sec').value)
        if cancel_after > 0.0:
            self._spin_for(cancel_after)
            self.stop()

        # 3) Sonucu bekle
        result_future = self._goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=30.0)
        if result_future.result() is None:
            self.get_logger().error('Sonuç zaman aşımına uğradı')
            return False
        status = result_future.result().status
        self.get_logger().info(
            f'Durum: {STATUS_NAMES.get(status, status)}, '
            f'dönülen açı: {result_future.result().result.delta:.2f} rad')
        return status == GoalStatus.STATUS_SUCCEEDED

    def stop(self) -> None:
        """Hedefi iptal eder ve hareket gerçekten bitene kadar bekler."""
        # 1) devam eden hedefi iptal et
        if self._goal_handle is not None:
            cancel_future = self._goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(
                self, cancel_future, timeout_sec=2.0)
            self.get_logger().info('İptal isteği gönderildi')

        # 2) hareket gerçekten duruncaya kadar bekle (en fazla 8 s)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 8.0:
            self._spin_for(0.1)
            if self._angular_speed < 0.01:      # rad/s
                break

        # 3) ancak şimdi kapatma adımlarına geçilebilir
        self.get_logger().info(
            f'Durdu (açısal hız {self._angular_speed:.3f} rad/s)')

    def _spin_for(self, duration_sec: float) -> None:
        """Düğümü verilen süre boyunca döndürür (geri çağırmalar işlensin)."""
        end = time.monotonic() + duration_sec
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)


def main(args: list[str] | None = None) -> None:
    """Bir dönüş hedefi gönderir ve sonucu yazdırır."""
    rclpy.init(args=args)
    node = RotateClient()
    try:
        node.rotate()
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
