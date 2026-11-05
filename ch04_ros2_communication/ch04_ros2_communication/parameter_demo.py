"""
Liste 12, 13 ve 15 - Parametre bildirimi, aralık sınırı ve çalışma anı değişikliği.

Düğüm, parametrelerine göre turtlesim kaplumbağasını çember çizdirerek
sürer: robot_name hangi kaplumbağanın sürüleceğini, publish_rate komut
frekansını, max_speed doğrusal hızı belirler. max_speed çalışma anında
değiştirildiğinde kaplumbağanın çemberi hemen büyür.

Çalıştırma:
    $ ros2 run turtlesim turtlesim_node
    $ cd ~/ros2_ws/src/ROS2-Book/ch04_ros2_communication
    $ ros2 run ch04_ros2_communication parameter_demo --ros-args \
        --params-file config/params.yaml
    $ ros2 param set /my_node max_speed 0.9     # kabul edilir
    $ ros2 param set /my_node max_speed 5.0     # reddedilir
"""

from geometry_msgs.msg import Twist
from rcl_interfaces.msg import (FloatingPointRange, ParameterDescriptor,
                                SetParametersResult)
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter


class ParameterDemo(Node):
    """Parametrelerle yönetilen basit bir hız komutu yayıncısı."""

    def __init__(self) -> None:
        super().__init__('my_node')

        # Verilen varsayılan değer aynı zamanda tipi belirler
        self.declare_parameter('robot_name', 'turtle1')      # string
        self.declare_parameter('publish_rate', 10)           # integer
        self.declare_parameter('logging_enabled', True)      # bool
        self.declare_parameter('operation_mode', 'force_on')  # string
        # Liste 13: açıklamalı ve aralık sınırlı bildirim
        self.declare_parameter(
            'max_speed', 0.5,
            ParameterDescriptor(
                description='Robotun azami doğrusal hızı [m/s]',
                floating_point_range=[FloatingPointRange(
                    from_value=0.0, to_value=1.0, step=0.01)]))

        rate_hz = self.get_parameter('publish_rate').value
        if rate_hz <= 0:
            raise ValueError(f'frekans pozitif olmalı: {rate_hz}')

        self._max_speed: float = self.get_parameter('max_speed').value
        robot_name = self.get_parameter('robot_name').value
        self._cmd_pub = self.create_publisher(
            Twist, f'/{robot_name}/cmd_vel', 10)

        # Liste 15: çalışma anındaki 'ros2 param set' çağrılarını doğrula
        self.add_on_set_parameters_callback(self._on_param_change)
        self._timer = self.create_timer(1.0 / rate_hz, self._tick)

        self.get_logger().info(
            f'robot_name={robot_name}, publish_rate={rate_hz} Hz, '
            f'max_speed={self._max_speed} m/s, operation_mode='
            f'{self.get_parameter("operation_mode").value}')

    def _tick(self) -> None:
        """Her periyotta güncel hız sınırıyla bir komut yayımlar."""
        cmd = Twist()
        cmd.linear.x = self._max_speed
        cmd.angular.z = 1.0
        self._cmd_pub.publish(cmd)
        if self.get_parameter('logging_enabled').value:
            # Her döngüde değil, en fazla 2 saniyede bir yaz
            self.get_logger().info(
                f'komut: {self._max_speed:.2f} m/s', throttle_duration_sec=2.0)

    def _on_param_change(
        self, params: list[Parameter]
    ) -> SetParametersResult:
        """Değişikliği doğrular; kabul edilirse davranışa yansıtır."""
        for param in params:
            if param.name == 'max_speed':
                if not (0.0 < param.value <= 1.0):
                    return SetParametersResult(
                        successful=False, reason='(0, 1] aralığında olmalı')
                self._max_speed = param.value   # kabul edildi
        return SetParametersResult(successful=True)


def main(args: list[str] | None = None) -> None:
    """Düğümü başlatır ve Ctrl+C gelene kadar döndürür."""
    rclpy.init(args=args)
    node = ParameterDemo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
