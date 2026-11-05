"""
Liste 20 - Xacro modelini koşum anında derleyip görüntüleme.

    $ ros2 launch ch04_ros2_communication mini_arm.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (Command, LaunchConfiguration,
                                  PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """robot_state_publisher, eklem kaydırıcısı ve RViz2'yi başlatır."""
    pkg_share = FindPackageShare('ch04_ros2_communication')
    model = LaunchConfiguration('model')
    # Command(['xacro ', model]): xacro komutu koşum anında çalışır ve
    # çıktısı (saf URDF metni) robot_description parametresine yazılır.
    robot_description = ParameterValue(
        Command(['xacro ', model]), value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument(
            'model',
            default_value=PathJoinSubstitution(
                [pkg_share, 'urdf', 'mini_arm.urdf.xacro']),
            description='Görüntülenecek Xacro dosyasının tam yolu'),
        Node(package='robot_state_publisher',
             executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description}]),
        # Eklem açılarını kaydırıcılarla elle değiştirmek için
        Node(package='joint_state_publisher_gui',
             executable='joint_state_publisher_gui'),
        Node(package='rviz2', executable='rviz2',
             arguments=['-d', PathJoinSubstitution(
                 [pkg_share, 'rviz', 'mini_arm.rviz'])]),
    ])
