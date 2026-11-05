"""
Liste 3 - İki tam sayıyı toplayan servis sunucusu.

Çalıştırma:
    $ ros2 run ch04_ros2_communication add_two_ints_server
"""

from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node


class AddTwoIntsServer(Node):
    """add_two_ints servisini sunan düğüm."""

    def __init__(self) -> None:
        super().__init__('add_two_ints_server')
        # create_service: arayüz tipi, servis adı ve geri çağırma fonksiyonu
        self._srv = self.create_service(
            AddTwoInts, 'add_two_ints', self._on_request)
        self.get_logger().info('Hazır: add_two_ints bekleniyor')

    def _on_request(
        self,
        request: AddTwoInts.Request,
        response: AddTwoInts.Response,
    ) -> AddTwoInts.Response:
        """İsteği işler ve doldurulmuş yanıtı döndürür."""
        response.sum = request.a + request.b
        self.get_logger().info(
            f'{request.a} + {request.b} = {response.sum}')
        # Yanıt mutlaka döndürülmelidir; döndürülmezse istemci süresiz bekler.
        return response


def main(args: list[str] | None = None) -> None:
    """Düğümü başlatır ve Ctrl+C gelene kadar döndürür."""
    rclpy.init(args=args)
    node = AddTwoIntsServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
