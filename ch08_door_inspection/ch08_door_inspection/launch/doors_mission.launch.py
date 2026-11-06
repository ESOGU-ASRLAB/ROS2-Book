"""
Kapı görevi: operatör paneliyle sürülen algılama + temizleme koşusu.

HARMONY senaryosundaki sensing_and_cleaning_mission.launch.py'nin kapı karşılığıdır ve
İKİ düğüm başlatır:

    doors_mission_node       -- /harmony/cmd_input'u bekler, ch08_door_inspection
                                başlatmasını yönetir, tur bitince senaryo kusurlarını
                                yayımlar
    cleaning_mission_runner  -- HARMONY senaryosundakiyle AYNIDIR: kusurları ve
                                CONFIRM'i bekler ve yalnızca UR'yi sürer

sensing_robot.py BİLEREK YOKTUR. Aynı /harmony/cmd_input'a abone olur ve START'ta kendi
elle yazılmış taramasını başlatırdı; UR iki yerden birden komut alırdı.
doors_mission_node onun yerini alır.

Temizleme sırasında Kawasaki özel bir işlem gerektirmez: cleaning_mission_runner ona
hiç komut vermez ve kapı muayenesi return_home_kawasaki:=true ile biter; yani temizleme
başladığında zaten evde park hâlindedir.

Beklenen sıra (panelin yaptığı budur):
    1. Robot hücresini başlat (araştırma deposu, harmony:=true)
    2. ros2 launch ch08_door_inspection doors_mission.launch.py
    3. /harmony/cmd_input üzerinde START yayımla   -> kapı muayenesi çalışır
    4. /harmony/cmd_input üzerinde CONFIRM yayımla -> temizleme çalışır

Panel olmadan kullanım:
    ros2 launch ch08_door_inspection doors_mission.launch.py only_sim:=false
    ros2 topic pub -1 /harmony/cmd_input std_msgs/msg/String "{data: '{"cmd":"START"}'}"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    mission_node = Node(
        package="ch08_door_inspection",
        executable="doors_mission_node",
        name="doors_mission_node",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "fixed_frame": "world",
            "only_sim": LaunchConfiguration("only_sim"),
            "force_replan": LaunchConfiguration("force_replan"),
            "extra_inspection_args": LaunchConfiguration("extra_inspection_args"),
            "defect_publish_interval": 0.3,
        }],
    )

    # HARMONY görev başlatmasıyla bayt bayt aynı yapılandırma: kusurlar ve gidilen eklem
    # konfigürasyonları aynıdır; bu yüzden burada kapıya özgü değiştirilecek
    # bir şey yoktur.
    cleaning_runner_node = Node(
        package="ch08_door_inspection",
        executable="cleaning_mission_runner",
        name="harmony_cleaning_runner",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "fixed_frame": "world",
            "real_robot_velocity": 0.1,
            "real_robot_acceleration": 0.1,
            "cleaning_duration": 3.0,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "only_sim", default_value="true",
            description="Forwarded to doors_inspection.launch.py: true -> capture only "
                        "the two Gazebo SICKs, false -> also the real ones. This is "
                        "about which CAMERAS are read, which is not the same question "
                        "as the HIL launch's use_fake_hardware (real ROBOT or not), "
                        "even though fake hardware only makes sense with only_sim:=true."),
        DeclareLaunchArgument(
            "force_replan", default_value="false",
            description="Forwarded to doors_inspection.launch.py. Cached trajectories "
                        "are replayed WITHOUT re-checking collision, so this must be "
                        "true for one run after any change to the collision model "
                        "(most recently the link7 gripper-pin box)."),
        DeclareLaunchArgument(
            "extra_inspection_args", default_value="",
            description="Extra 'name:=value' arguments appended to the ch08_door_inspection "
                        "launch, e.g. \"kawasaki_velocity:=0.05\". Escape hatch so the "
                        "orchestrator does not need a parameter per knob."),
        mission_node,
        cleaning_runner_node,
    ])
