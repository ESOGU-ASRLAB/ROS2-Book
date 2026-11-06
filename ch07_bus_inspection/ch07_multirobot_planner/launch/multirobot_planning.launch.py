"""
Bölüm 7 - 1. Aşama: çevrim dışı çok robotlu bakış noktası planlaması.

Şase ağından bakış noktaları üretir, iki kola paylaştırır ve planı JSON'a yazar.
MoveIt 2'nin /compute_ik servisi için robot hücresinin (araştırma deposu)
ayakta olması gerekir.

    $ ros2 launch ch07_multirobot_planner multirobot_planning.launch.py
    $ ros2 service call /multirobot_planner_node/plan std_srvs/srv/Trigger
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ch07_multirobot_planner import paths


def generate_launch_description() -> LaunchDescription:
    """Planlayıcı düğümünü ve plan görselleştiricisini başlatır."""
    default_config = os.path.join(
        get_package_share_directory('ch07_multirobot_planner'),
        'config', 'multirobot_params.yaml')

    mesh_path_arg = DeclareLaunchArgument(
        'mesh_path',
        default_value=paths.chassis_mesh_file(),
        description='Path to the chassis.stl file (default: ur_description package)')
    coverage_threshold_arg = DeclareLaunchArgument(
        'coverage_threshold', default_value='0.98',
        description='Target coverage threshold (0.0 to 1.0)')
    config_file_arg = DeclareLaunchArgument(
        'config_file', default_value=default_config,
        description='Camera/reachability/allocation parameters yaml')
    output_plan_file_arg = DeclareLaunchArgument(
        'output_plan_file',
        default_value=paths.output_path('multirobot_viewpoint_plan.json'),
        description='Where to write the generated multi-robot plan JSON')

    planner_node = Node(
        package='ch07_multirobot_planner',
        executable='multirobot_planner_node',
        name='multirobot_planner_node',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'mesh_path': LaunchConfiguration('mesh_path'),
                'coverage_threshold': LaunchConfiguration('coverage_threshold'),
                'output_plan_file': LaunchConfiguration('output_plan_file'),
            },
        ],
    )

    visualizer_node = Node(
        package='ch07_multirobot_planner',
        executable='multirobot_viewpoint_visualizer',
        name='multirobot_viewpoint_visualizer',
        output='screen',
        parameters=[{
            'plan_file': LaunchConfiguration('output_plan_file'),
        }],
    )

    return LaunchDescription([
        mesh_path_arg,
        coverage_threshold_arg,
        config_file_arg,
        output_plan_file_arg,
        planner_node,
        visualizer_node,
    ])
