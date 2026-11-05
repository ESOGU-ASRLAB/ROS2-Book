"""
Liste 22 - Bir noktanın başka bir çerçeveye çevrilmesi (Buffer ve TransformListener).

camera_point konusundan gelen PointStamped mesajını (örneğin kameranın
optik çerçevesinde ölçülmüş bir nesne) target_frame çerçevesine çevirir.

Çalıştırma (mini_arm modeli açıkken):
    $ ros2 launch ch04_ros2_communication mini_arm.launch.py
    $ ros2 run ch04_ros2_communication point_transformer
    $ ros2 topic pub --once /camera_point geometry_msgs/msg/PointStamped \
        "{header: {frame_id: camera_optical_frame}, point: {x: 0.0, y: 0.0, z: 0.5}}"
"""

from geometry_msgs.msg import PointStamped
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener
# Import edilmesi PointStamped çevirisini Buffer'a kaydeder; kullanılmıyor
# gibi görünse de kaldırılamaz.
import tf2_geometry_msgs  # noqa: F401


class PointTransformer(Node):
    """Gelen noktaları hedef çerçeveye çevirip günlüğe yazan düğüm."""

    def __init__(self) -> None:
        super().__init__('point_transformer')
        self.declare_parameter('target_frame', 'base_link')
        self._target_frame: str = self.get_parameter('target_frame').value
        # Buffer dönüşümleri saklar; TransformListener /tf ve /tf_static'i
        # dinleyip Buffer'ı doldurur.
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self)
        self.create_subscription(
            PointStamped, 'camera_point', self._on_point, 10)

    def _on_point(self, msg: PointStamped) -> None:
        """Noktayı hedef çerçeveye çevirir; dönüşüm yoksa uyarır."""
        try:
            result = self._buffer.transform(
                msg, self._target_frame, timeout=Duration(seconds=0.5))
        except TransformException as exc:
            self.get_logger().warn(f'dönüşüm yok: {exc}')
            return
        self.get_logger().info(
            f'{msg.header.frame_id} -> {self._target_frame}: '
            f'({result.point.x:.3f}, {result.point.y:.3f}, '
            f'{result.point.z:.3f})')


def main(args: list[str] | None = None) -> None:
    """Düğümü başlatır ve Ctrl+C gelene kadar döndürür."""
    rclpy.init(args=args)
    node = PointTransformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
