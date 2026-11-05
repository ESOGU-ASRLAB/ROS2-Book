"""
Liste 16 - Argüman, ad alanı ve düğüm içeren başlatma dosyası.

İki ayrı turtlesim penceresi açar; birincisinin arka plan rengi dışarıdan
verilebilir.

    $ ros2 launch ch04_ros2_communication two_turtles.launch.py --show-args
    $ ros2 launch ch04_ros2_communication two_turtles.launch.py background_r:=200
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """İki turtlesim düğümünü farklı ad alanlarında başlatır."""
    background_arg = DeclareLaunchArgument(
        'background_r', default_value='69',
        description='Pencere arka planının kırmızı bileşeni (0-255)')

    # Ad alanları (sim1, sim2) sayesinde aynı adlı iki düğüm çakışmaz:
    # /sim1/sim ve /sim2/sim olarak ayrışırlar.
    turtle1 = Node(
        package='turtlesim', executable='turtlesim_node',
        namespace='sim1', name='sim', output='screen',
        # Argümanın metin değeri ('200') YAML kurallarıyla okunur ve
        # tamsayıya dönüşür; turtlesim'in beklediği tip de budur.
        parameters=[{'background_r': LaunchConfiguration('background_r')}])

    turtle2 = Node(
        package='turtlesim', executable='turtlesim_node',
        namespace='sim2', name='sim', output='screen')

    return LaunchDescription([background_arg, turtle1, turtle2])
