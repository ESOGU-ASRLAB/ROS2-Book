"""
Alt başlatma dosyası - 6-eksenli kolun modelini ve TF ağacını yayımlar.

Liste 26'da çalıştırılan ve Liste 18'deki cell_bringup.launch.py tarafından
dâhil edilen dosyadır. Tek başına çalıştırıldığında kendi varsayılanlarını
kullanır; üst dosyadan çağrıldığında use_sim_time ve tcp_offset_z üstten
gelen değerlerle ezilir.

    $ ros2 launch ch04_ros2_communication manipulator_rsp.launch.py
    $ ros2 launch ch04_ros2_communication manipulator_rsp.launch.py \
        tcp_offset_z:=0.220 gui:=true rviz:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (Command, LaunchConfiguration,
                                  PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """robot_state_publisher ve eklem durumu yayıncısını başlatır."""
    pkg_share = FindPackageShare('ch04_ros2_communication')
    use_sim_time = LaunchConfiguration('use_sim_time')
    gui = LaunchConfiguration('gui')

    # tcp_offset_z, Xacro'daki <xacro:arg name="tcp_offset_z"> değerini ezer
    robot_description = ParameterValue(Command([
        'xacro ',
        PathJoinSubstitution([pkg_share, 'urdf', 'six_axis_arm.urdf.xacro']),
        ' tcp_offset_z:=', LaunchConfiguration('tcp_offset_z'),
    ]), value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Simülasyon saati (/clock) kullanılsın mı?'),
        DeclareLaunchArgument(
            'tcp_offset_z', default_value='0.145',
            description='Flanştan (tool0) TCP ucuna Z ekseni ofseti [m]'),
        DeclareLaunchArgument(
            'gui', default_value='false',
            description='true: eklemleri kaydırıcılarla oynat'),
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description='true: RViz2 ile TF ağacını göster'),

        Node(package='robot_state_publisher',
             executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description,
                          'use_sim_time': use_sim_time}]),
        # Eklem açılarını yayımlayan biri olmadan robot_state_publisher
        # yalnızca sabit (fixed) eklemleri yayımlar; hareketli zincir eksik
        # kalır. GUI kapalıyken tüm eklemler sıfır açıda yayımlanır.
        Node(package='joint_state_publisher',
             executable='joint_state_publisher',
             parameters=[{'use_sim_time': use_sim_time}],
             condition=UnlessCondition(gui)),
        Node(package='joint_state_publisher_gui',
             executable='joint_state_publisher_gui',
             parameters=[{'use_sim_time': use_sim_time}],
             condition=IfCondition(gui)),
        Node(package='rviz2', executable='rviz2',
             arguments=['-d', PathJoinSubstitution(
                 [pkg_share, 'rviz', 'six_axis_arm.rviz'])],
             parameters=[{'use_sim_time': use_sim_time}],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
