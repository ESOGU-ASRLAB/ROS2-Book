"""Bölüm 10 - İnsan-robot iş birlikli vidalama senaryosunun başlatma dosyası."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    # --- GPIO parametreleri (pendant'taki GPIO sekmesinin karşılığı) ---
    # IFARLAB hücresinde find_io_pins.py ile ölçülen pin haritası:
    #   DOUT0 -> vidalama SIKMA      DIN7 -> YEŞİL buton
    #   DOUT1 -> vidalama SÖKME      DIN6 -> KIRMIZI buton
    #                                DIN5 -> BEYAZ buton
    declared_arguments = [
        DeclareLaunchArgument(
            "screwdriver_pin",
            default_value="0",
            description="Vidalama SIKMA çıkışı (DOUT0).",
        ),
        DeclareLaunchArgument(
            "green_button_pin",
            default_value="7",
            description="Operatörün yeşil butonu (DIN7).",
        ),
        DeclareLaunchArgument(
            "green_button_active_high",
            default_value="true",
            description="Buton NO ise true, NC ise false.",
        ),
        DeclareLaunchArgument(
            "green_button_timeout",
            default_value="0.0",
            description="Yeşil buton bekleme zaman aşımı (sn). 0.0 = süresiz bekle.",
        ),
        DeclareLaunchArgument(
            "travel_speed", default_value="0.1",
            description="Noktalar arası normal seyir hız/ivme ölçeği (0.0-1.0).",
        ),
        DeclareLaunchArgument(
            "screw_speed", default_value="0.01",
            description="xTop <-> xOpt vidalama geçişlerinin hız/ivme ölçeği (0.0-1.0).",
        ),
        DeclareLaunchArgument(
            "planner_id",
            default_value="RRTConnectkConfigDefault",
            description=(
                "OMPL planlayıcı. RRTstarkConfigDefault optimize edicidir ve verilen "
                "sürenin tamamını kullanır; hızlı çevrim için RRTConnect uygundur."
            ),
        ),
        DeclareLaunchArgument(
            "planning_time", default_value="5.0",
            description="Plan başına izin verilen süre (sn).",
        ),
        DeclareLaunchArgument(
            "planning_attempts", default_value="10",
            description="Plan başına deneme sayısı.",
        ),
        DeclareLaunchArgument(
            "gpio_mode",
            default_value="auto",
            description=(
                "auto: use_fake_hardware / ENV_USE_FAKE_HARDWARE ve io_states yayınına "
                "bakarak kendisi karar verir. force_on: zorla açık. force_off: zorla kapalı."
            ),
        ),
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description=(
                "Sürücüyü hangi modda başlattıysanız aynısını verin. true ise GPIO "
                "devre dışı bırakılır: mock_components dijital girişleri hep 0 döndürdüğü "
                "için yeşil buton beklemesi sonsuza kadar takılırdı."
            ),
        ),
        DeclareLaunchArgument(
            "fake_button_delay",
            default_value="0.0",
            description="GPIO kapalıyken yeşil buton yerine beklenecek süre (sn).",
        ),
        DeclareLaunchArgument(
            "trajectory_enable", default_value="true",
            description=(
                "true: kol hareketleri ilk seferde planlanıp JSON'a kaydedilir, sonra "
                "hep aynısı oynatılır. false: her hareket yeniden planlanır."
            ),
        ),
        DeclareLaunchArgument(
            "trajectory_dir", default_value="",
            description=(
                "Kayıt klasörü (<dir>/hrc_screwing.json). Boş: "
                "~/ros2_book_output/ch10/trajectories. Dosya silinirse hepsi yeniden "
                "kaydedilir."
            ),
        ),
    ]

    human_robot_collaboration_node = Node(
        package="ch10_screwdriving",
        executable="human_robot_collaboration_scenario",
        name="human_robot_collaboration_scenario",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                # LaunchConfiguration'lar string döner; parametreler int/bool/float
                # olarak deklare edildiği için tip dönüşümü zorunlu.
                "screwdriver_pin": ParameterValue(
                    LaunchConfiguration("screwdriver_pin"), value_type=int
                ),
                "green_button_pin": ParameterValue(
                    LaunchConfiguration("green_button_pin"), value_type=int
                ),
                "green_button_active_high": ParameterValue(
                    LaunchConfiguration("green_button_active_high"), value_type=bool
                ),
                "green_button_timeout": ParameterValue(
                    LaunchConfiguration("green_button_timeout"), value_type=float
                ),
                # value_type=str şart: aksi halde "off"/"on" YAML boolean'a
                # dönüşür ve string parametreye atanamayıp node düşer.
                "travel_speed": ParameterValue(
                    LaunchConfiguration("travel_speed"), value_type=float
                ),
                "screw_speed": ParameterValue(
                    LaunchConfiguration("screw_speed"), value_type=float
                ),
                "planner_id": ParameterValue(
                    LaunchConfiguration("planner_id"), value_type=str
                ),
                "planning_time": ParameterValue(
                    LaunchConfiguration("planning_time"), value_type=float
                ),
                "planning_attempts": ParameterValue(
                    LaunchConfiguration("planning_attempts"), value_type=int
                ),
                "gpio_mode": ParameterValue(
                    LaunchConfiguration("gpio_mode"), value_type=str
                ),
                "use_fake_hardware": ParameterValue(
                    LaunchConfiguration("use_fake_hardware"), value_type=bool
                ),
                "fake_button_delay": ParameterValue(
                    LaunchConfiguration("fake_button_delay"), value_type=float
                ),
                "trajectory_enable": ParameterValue(
                    LaunchConfiguration("trajectory_enable"), value_type=bool
                ),
                "trajectory_dir": ParameterValue(
                    LaunchConfiguration("trajectory_dir"), value_type=str
                ),
            }
        ],
        # Senaryo tek tur koşup çıkar; düğüm bittiğinde launch da kapansın.
        on_exit=Shutdown(),
    )

    return LaunchDescription(declared_arguments + [human_robot_collaboration_node])
