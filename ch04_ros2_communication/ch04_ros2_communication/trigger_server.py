"""
Liste 6 - Hatanın yanıta taşındığı bir Trigger sunucusu.

Servis fonksiyonu içindeki bir istisna istemciye taşınmaz; yalnızca sunucu
günlüğüne düşer. Bu yüzden başarı bilgisi yanıtın success/message
alanlarıyla döndürülür.

Çalıştırma:
    $ ros2 run ch04_ros2_communication trigger_server
    $ ros2 service call /start std_srvs/srv/Trigger
    # Hata yolunu denemek için:
    $ ros2 param set /trigger_server simulate_failure true
    $ ros2 service call /start std_srvs/srv/Trigger
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class TriggerServer(Node):
    """Başarısız olabilen bir işi Trigger servisiyle sunan düğüm."""

    def __init__(self) -> None:
        super().__init__('trigger_server')
        # Hata yolunu elle tetiklemek için bir bayrak
        self.declare_parameter('simulate_failure', False)
        self._processed_count = 0
        self._srv = self.create_service(Trigger, 'start', self._on_start)
        self.get_logger().info('Hazır: /start bekleniyor')

    def _do_work(self) -> int:
        """Başarısız olabilen örnek iş; işlenen toplam öğe sayısını döndürür."""
        if self.get_parameter('simulate_failure').value:
            raise RuntimeError('sensör yanıt vermedi (benzetim)')
        self._processed_count += 3
        return self._processed_count

    def _on_start(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        """İşi çalıştırır; sonucu ya da hatayı yanıta yazar."""
        try:
            count = self._do_work()          # başarısız olabilen iş
            response.success = True
            response.message = f'{count} öğe işlendi'
        except Exception as exc:             # istisna yanıta taşınmalıdır
            response.success = False
            response.message = f'başarısız: {exc}'
            self.get_logger().error(response.message)
        return response


def main(args: list[str] | None = None) -> None:
    """Düğümü başlatır ve Ctrl+C gelene kadar döndürür."""
    rclpy.init(args=args)
    node = TriggerServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
