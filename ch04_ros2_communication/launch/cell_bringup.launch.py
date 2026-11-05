"""
Liste 18 - Üst başlatma dosyasından alt dosyaları dâhil etme ve argüman ezme.

Gazebo'yu (ros_gz_sim) ve manipulator_rsp.launch.py dosyasını
IncludeLaunchDescription ile birleştirir. Alt dosyanın use_sim_time ve
tcp_offset_z varsayılanları buradaki değerlerle ezilir.

    $ ros2 launch ch04_ros2_communication cell_bringup.launch.py
    $ ros2 launch ch04_ros2_communication cell_bringup.launch.py tcp_offset_z:=0.220
    # Gazebo olmadan yalnızca argüman ezmeyi denemek için:
    $ ros2 launch ch04_ros2_communication cell_bringup.launch.py \
        use_gazebo:=false use_sim_time:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    """Gazebo ve robot modelini tek komutla başlatır."""
    # 1) Üst seviye argümanların tanımlanması
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Simülasyon saati kullanılsın mı?')
    tcp_offset_arg = DeclareLaunchArgument(
        'tcp_offset_z', default_value='0.145',
        description='Flanştan (tool0) TCP ucuna Z ekseni ofseti [m]')
    use_gazebo_arg = DeclareLaunchArgument(
        'use_gazebo', default_value='true',
        description='false: Gazebo açılmaz, yalnızca robot modeli yayımlanır')

    # 2) Alt başlatma dosyası 1: Gazebo simülasyonu (argüman ezme ile).
    #    Bu bölümde robot Gazebo'ya eklenmez; amaç dâhil etmeyi göstermektir.
    #    Robotun Gazebo'da çalıştırılması Bölüm 5 paketindedir.
    gz_launch_path = os.path.join(
        get_package_share_directory('ros_gz_sim'),
        'launch', 'gz_sim.launch.py')
    gazebo_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_launch_path),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
        condition=IfCondition(LaunchConfiguration('use_gazebo')),
    )

    # 3) Alt başlatma dosyası 2: Manipülatör RSP ve TCP kurulumu.
    #    Alt dosyadaki 'use_sim_time' ve 'tcp_offset_z' varsayılanları
    #    üstteki değerlerle ezilir.
    robot_launch_path = os.path.join(
        get_package_share_directory('ch04_ros2_communication'),
        'launch', 'manipulator_rsp.launch.py')
    robot_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(robot_launch_path),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'tcp_offset_z': LaunchConfiguration('tcp_offset_z'),
        }.items(),
    )

    return LaunchDescription([
        use_sim_time_arg,
        tcp_offset_arg,
        use_gazebo_arg,
        gazebo_include,
        robot_include,
    ])
