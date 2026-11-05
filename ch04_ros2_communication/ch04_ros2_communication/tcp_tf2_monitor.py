"""
Liste 25 - 6-eksenli kolda base_link -> tool0 ve base_link -> tcp_link dönüşümleri.

Her saniye bilek flanşının (tool0) ve takım ucunun (tcp_link) robot
tabanına göre konumunu sorgular; ayrıca TCP'nin 5 cm önündeki bir numune
noktasını base_link çerçevesine çevirir.

Çalıştırma:
    $ ros2 launch ch04_ros2_communication manipulator_rsp.launch.py
    $ ros2 run ch04_ros2_communication tcp_tf2_monitor
"""

from geometry_msgs.msg import PointStamped
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
import tf2_geometry_msgs  # noqa: F401  PointStamped dönüşümü için zorunlu kayıt


class ManipulatorTcpMonitor(Node):
    """Flanş ve TCP çerçevelerini TF2 üzerinden izleyen düğüm."""

    def __init__(self) -> None:
        super().__init__('manipulator_tcp_monitor')
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._timer = self.create_timer(1.0, self._compute_flange_and_tcp)
        self.get_logger().info(
            '6-Eksenli Manipülatör Flanş/TCP TF2 izleyici başlatıldı.')

    def _compute_flange_and_tcp(self) -> None:
        """Flanş/TCP konumlarını ve numune noktasını günlüğe yazar."""
        try:
            # 1) Tabandan (base_link) bilek flanşına (tool0) dönüşüm.
            #    Time() = "elimdeki en güncel dönüşüm".
            t_flange = self._tf_buffer.lookup_transform(
                'base_link', 'tool0', Time(),
                timeout=Duration(seconds=0.2))

            # 2) Tabandan (base_link) takım ucuna (tcp_link) bileşik dönüşüm
            t_tcp = self._tf_buffer.lookup_transform(
                'base_link', 'tcp_link', Time(),
                timeout=Duration(seconds=0.2))

            fl_p = t_flange.transform.translation
            tcp_p = t_tcp.transform.translation
            tcp_q = t_tcp.transform.rotation

            self.get_logger().info(
                f'Flanş (tool0) [m]: X={fl_p.x:.3f}, Y={fl_p.y:.3f}, '
                f'Z={fl_p.z:.3f} | TCP (tcp_link) [m]: X={tcp_p.x:.3f}, '
                f'Y={tcp_p.y:.3f}, Z={tcp_p.z:.3f} (Quat w={tcp_q.w:.2f})')

            # 3) TCP ucunun 5 cm önünde algılanan bir yüzey noktasının
            #    base_link karşılığı (TCP'nin +Z ekseni takımın baktığı yön)
            sample_tcp = PointStamped()
            sample_tcp.header.frame_id = 'tcp_link'
            sample_tcp.header.stamp = Time().to_msg()
            sample_tcp.point.z = 0.050   # TCP'den 5 cm ilerideki numune

            sample_base = self._tf_buffer.transform(
                sample_tcp, 'base_link', timeout=Duration(seconds=0.2))
            self.get_logger().info(
                'Numune Noktası (base_link) [m]: '
                f'({sample_base.point.x:.3f}, {sample_base.point.y:.3f}, '
                f'{sample_base.point.z:.3f})')

        except TransformException as exc:
            self.get_logger().warn(f'TF2 ağacı henüz hazır değil: {exc}')


def main(args: list[str] | None = None) -> None:
    """Düğümü başlatır ve Ctrl+C gelene kadar döndürür."""
    rclpy.init(args=args)
    node = ManipulatorTcpMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
