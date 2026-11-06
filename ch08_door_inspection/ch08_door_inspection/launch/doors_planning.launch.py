"""
İKİ araba kapısı üzerinde ortak bir kaplama turu planlar ve onu JSON'a yazar.

Varsayılan çıktı: ~/ros2_book_output/ch08/doors_viewpoint_plan.json.

Şase işiyle AYNI multirobot_planner_node'u çalıştırır; düğüm zaten ağdan bağımsızdır.
Buradaki tek kapıya özgü şeyler birleştirilmiş hedef ağ, mesh_scale=1.0 (kapı STL'leri
milimetre değil metredir) ve bu paketin parametre dosyasıdır.

Düğüm ayağa kalkınca servisi çağırın:
    ros2 service call /doors_planner_node/plan std_srvs/srv/Trigger
"""
import os
from typing import Any, List

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ch08_door_inspection.merge_door_meshes import DEFAULT_DOORS, DEFAULT_OUTPUT, ensure_merged

from ch08_door_inspection import paths


def _launch_setup(context: LaunchContext, *args: Any, **kwargs: Any) -> List[Node]:
    # Birleştirilmiş hedef ağı planlayıcı başlamadan ÖNCE kur. Ayrı bir süreç yerine
    # burada yapılır; böylece sıra güvence altındadır: birleştirmenin bitmesiyle
    # planlayıcının dosyayı okuması arasında yarış olmaz. Bir kaynak STL değişmedikçe
    # etkisizdir (yalnızca bir stat çağrısı).
    merged = ensure_merged(
        sources=[p for p in LaunchConfiguration('doors').perform(context).split(',') if p],
        output=LaunchConfiguration('merged_mesh').perform(context),
        force=LaunchConfiguration('force_merge').perform(context).lower() in ('true', '1'),
    )

    planner_node = Node(
        package='ch07_multirobot_planner',
        executable='multirobot_planner_node',
        name='doors_planner_node',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'mesh_path': merged,
                # KAPIYA ÖZGÜ değer. windowdoor.stl / windowlessdoor.stl zaten metre
                # cinsindendir (URDF onları scale="1 1 1" ile yükler);
                # milimetre olan chassis.stl'den farklı.
                'mesh_scale': 1.0,
                'coverage_threshold': LaunchConfiguration('coverage_threshold'),
                'output_plan_file': LaunchConfiguration('output_plan_file'),
            },
        ],
    )

    visualizer_node = Node(
        package='ch07_multirobot_planner',
        executable='multirobot_viewpoint_visualizer',
        name='doors_viewpoint_visualizer',
        output='screen',
        parameters=[{'plan_file': LaunchConfiguration('output_plan_file')}],
    )

    return [planner_node, visualizer_node]


def generate_launch_description() -> LaunchDescription:
    default_config = os.path.join(
        get_package_share_directory('ch08_door_inspection'), 'config', 'doors_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'doors', default_value=','.join(DEFAULT_DOORS),
            description='Comma-separated source door STLs merged into one target mesh.'),
        DeclareLaunchArgument(
            'merged_mesh', default_value=DEFAULT_OUTPUT,
            description='Where the merged door mesh is cached (a build artefact, ~45 MB, '
                        'rebuilt automatically whenever a source STL is newer).'),
        DeclareLaunchArgument(
            'force_merge', default_value='false',
            description='true -> rebuild the merged mesh even if it looks up to date.'),
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='Camera / reachability / allocation parameters.'),
        DeclareLaunchArgument(
            'coverage_threshold', default_value='0.98',
            description='Stop once 98% of the reachable targets are covered. Lowered '
                        'from 0.992 after the first real run: with the Kawasaki capped '
                        'at 3 viewpoints the last tour only reached 96.84%, so 0.992 '
                        'never fired and the tail cutter ended the run instead. Passed '
                        'AFTER the config file and therefore overrides it.'),
        DeclareLaunchArgument(
            'output_plan_file',
            default_value=paths.output_path('doors_viewpoint_plan.json'),
            description='Where to write the generated doors plan JSON.'),
        OpaqueFunction(function=_launch_setup),
    ])
