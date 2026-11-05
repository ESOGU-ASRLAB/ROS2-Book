"""
Liste 5b - Bringup başlatma dosyasından kip yönetimi ve argüman aktarımı.

Tek bir use_fake_hardware argümanı tüm sistemin hangi kipte kalkacağını
belirler; aynı değer Xacro'ya da aktarılır (Liste 5a):

    # Kip 1: Gazebo Harmonic + gz_ros2_control (fizik, kamera, /clock)
    $ ros2 launch ch05_simulation_visualization cell_bringup.launch.py

    # Kip 2: Donanımsız saf yazılım testi (mock_components/GenericSystem)
    $ ros2 launch ch05_simulation_visualization cell_bringup.launch.py \
        use_fake_hardware:=true

Ardından kolu hareket ettirmek için (Gazebo kipinde use_sim_time:=true):
    $ ros2 run ch05_simulation_visualization send_trajectory
"""

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, RegisterEventHandler)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, LaunchConfiguration,
                                  PathJoinSubstitution, PythonExpression)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Robot modelini seçilen donanım kipinde ve RViz2 ile başlatır."""
    pkg_share = FindPackageShare('ch05_simulation_visualization')
    use_fake_hw = LaunchConfiguration('use_fake_hardware')

    # Tek bir parametre ile simülasyon veya sahte donanım seçimi
    use_fake_hw_arg = DeclareLaunchArgument(
        'use_fake_hardware', default_value='false',
        description='true ise Mock/GenericSystem, false ise Gazebo sürücüsü')
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true', description='RViz2 açılsın mı?')
    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='true: Gazebo arayüzsüz (yalnız sunucu) çalışır')

    # Simülasyon saati yalnızca Gazebo kipinde vardır. Mock kipinde
    # use_sim_time açık kalırsa /clock hiç gelmez ve düğümler donar.
    use_sim_time = PythonExpression(["'", use_fake_hw, "' != 'true'"])

    # Xacro komutuna argümanın iletilmesi
    xacro_path = PathJoinSubstitution(
        [pkg_share, 'urdf', 'manipulator.urdf.xacro'])
    robot_description = ParameterValue(
        Command(['xacro ', xacro_path,
                 ' use_fake_hardware:=', use_fake_hw]),
        value_type=str)
    controllers_yaml = PathJoinSubstitution(
        [pkg_share, 'config', 'controllers.yaml'])

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description,
                     'use_sim_time': use_sim_time}])

    def spawner(name: str) -> Node:
        """Verilen denetleyiciyi yükleyip etkinleştiren spawner düğümü."""
        return Node(package='controller_manager', executable='spawner',
                    arguments=[name, '--controller-manager',
                               '/controller_manager'],
                    parameters=[{'use_sim_time': use_sim_time}])

    # --- Kip 2: Mock donanım --------------------------------------------
    # Denetleyici yöneticisini ros2_control_node ayrı bir süreç olarak çalıştırır.
    mock_group = GroupAction(
        condition=IfCondition(use_fake_hw),
        actions=[
            Node(package='controller_manager',
                 executable='ros2_control_node',
                 parameters=[{'robot_description': robot_description},
                             controllers_yaml],
                 output='screen'),
            spawner('joint_state_broadcaster'),
            spawner('joint_trajectory_controller'),
        ])

    # --- Kip 1: Gazebo Harmonic -----------------------------------------
    # Denetleyici yöneticisi Gazebo'nun içinde (gz_ros2_control eklentisi)
    # çalışır; robot dünyaya eklenmeden denetleyiciler yüklenemez.
    gz_args = PythonExpression([
        "'-r -v 1 ' + ('-s ' if '", LaunchConfiguration('headless'),
        "' == 'true' else '') + '",
        PathJoinSubstitution([pkg_share, 'worlds', 'cell.sdf']), "'"])
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])),
        launch_arguments={'gz_args': gz_args}.items())

    # robot_description konusundaki modeli Gazebo dünyasına ekler ve çıkar
    spawn_robot = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-topic', 'robot_description',
                   '-name', 'manipulator', '-allow_renaming', 'true'])

    # Liste 6: konular config/bridge.yaml dosyasında tanımlıdır
    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        parameters=[{
            'config_file': PathJoinSubstitution(
                [pkg_share, 'config', 'bridge.yaml']),
            'use_sim_time': True,
        }],
        output='screen')

    jsb_spawner = spawner('joint_state_broadcaster')
    jtc_spawner = spawner('joint_trajectory_controller')
    gazebo_group = GroupAction(
        condition=UnlessCondition(use_fake_hw),
        actions=[
            gazebo,
            bridge,
            spawn_robot,
            # Sıralama sabit bir gecikmeyle değil olaylarla kurulur
            # (Bölüm 4.4.2): robot eklenip 'create' çıkınca durum
            # yayıncısı, o da yüklenince yörünge denetleyicisi başlar.
            RegisterEventHandler(OnProcessExit(
                target_action=spawn_robot, on_exit=[jsb_spawner])),
            RegisterEventHandler(OnProcessExit(
                target_action=jsb_spawner, on_exit=[jtc_spawner])),
        ])

    rviz = Node(
        package='rviz2', executable='rviz2',
        arguments=['-d', PathJoinSubstitution([pkg_share, 'rviz', 'cell.rviz'])],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('rviz')))

    return LaunchDescription([
        use_fake_hw_arg,
        rviz_arg,
        headless_arg,
        robot_state_publisher,
        mock_group,
        gazebo_group,
        rviz,
    ])
