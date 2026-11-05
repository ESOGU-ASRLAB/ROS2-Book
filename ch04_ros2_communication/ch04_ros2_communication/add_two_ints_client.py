"""
Liste 4 - add_two_ints servisini çağıran istemci düğümü.

Çalıştırma (sunucu ayaktayken):
    $ ros2 run ch04_ros2_communication add_two_ints_client
    $ ros2 run ch04_ros2_communication add_two_ints_client 3 4
"""

import sys

from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node


class AddTwoIntsClient(Node):
    """Servis istemcisi: iki sayıyı sunucuya gönderip toplamı alır."""

    def __init__(self) -> None:
        super().__init__('add_two_ints_client')
        self._cli = self.create_client(AddTwoInts, 'add_two_ints')
        # Sunucu ayağa kalkana kadar bekle; yoksa istek boşa gider.
        while not self._cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Sunucu bekleniyor...')

    def add(self, a: int, b: int) -> int:
        """İsteği gönderir ve yanıt gelene kadar düğümü döndürür."""
        request = AddTwoInts.Request()
        request.a, request.b = a, b
        # call_async bir 'gelecek' (future) döndürür; yanıt ancak düğüm
        # döndürüldüğünde işlenir. Bu yüzden spin_until_future_complete.
        future = self._cli.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result().sum


def main(args: list[str] | None = None) -> None:
    """Komut satırındaki iki sayıyı (varsayılan 7 ve 5) toplatır."""
    rclpy.init(args=args)
    # ROS argümanları ayıklandıktan sonra kalan konum argümanları
    user_args = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    a, b = (int(user_args[0]), int(user_args[1])) if len(user_args) >= 2 \
        else (7, 5)
    node = AddTwoIntsClient()
    print(f'{a} + {b} =', node.add(a, b))
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
