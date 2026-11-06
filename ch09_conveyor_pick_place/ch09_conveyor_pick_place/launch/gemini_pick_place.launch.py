#!/usr/bin/env python3
"""
Gemini pick and place düğümlerini başlatır.

Simülasyonu başlatmaz. Önce hücreyi ayağa kaldırın:
    ros2 launch my_robot_cell_control hil_test_whole_unified.launch.py

sonra:
    ros2 launch ch09_conveyor_pick_place gemini_pick_place.launch.py            # sim
    ros2 launch ch09_conveyor_pick_place gemini_pick_place.launch.py mode:=real # gerçek

TEK argüman vardır: mode. Sim ile gerçek arasındaki tek fark KAMERA (ve saat)
olduğu için başka bir şeyin argüman olmasına gerek yok:

  mode:=sim  -> config/mode_sim.yaml    (/sim/... köprü topic'leri, use_sim_time)
  mode:=real -> config/mode_real.yaml   (SICK sürücü topic'leri, gerçek zaman)

MoveIt tarafı ikisinde de AYNI: komutlar /move_action'a, yani gerçek MoveIt'e
gider. Gazebo'daki robot real_to_sim_bridge sayesinde onu aynalar. Gerçek
robota geçerken hareket tarafında değişecek hiçbir şey yok.

Geri kalan her ayar config/gemini_params.yaml içinde. Tek seferlik denemeler
için o dosyayı düzenleyin ya da:
    ros2 launch ... gemini_pick_place.launch.py --ros-args -p dry_run:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    config_dir = os.path.join(
        get_package_share_directory("ch09_conveyor_pick_place"), "config"
    )

    base_params = os.path.join(config_dir, "gemini_params.yaml")

    # mode_sim.yaml / mode_real.yaml -- dosya adı mode'dan kurulur, böylece
    # yeni bir mod eklemek için launch'a dokunmak gerekmez.
    mode_params = PathJoinSubstitution(
        [config_dir, ["mode_", LaunchConfiguration("mode"), ".yaml"]]
    )

    # SIRA ÖNEMLİ: mode overlay'i temel dosyanın üstüne biner.
    parameters = [base_params, mode_params]

    return LaunchDescription([
        DeclareLaunchArgument(
            "mode", default_value="sim", choices=["sim", "real"],
            description="Kamera kaynağı: sim (Gazebo köprüsü) | real (SICK sürücüsü). "
                        "MoveIt her iki modda da gerçek /move_action'ı kullanır.",
        ),
        DeclareLaunchArgument(
            "record", default_value="true", choices=["true", "false"],
            description="Koşuyu ~/ros2_book_output/ch09/recordings/<zaman damgası>/ "
                        "altına kaydet. Kayıt düğümü pasiftir: yalnız dinler, hiçbir şey "
                        "komutlamaz; kapatmak için record:=false.",
        ),
        DeclareLaunchArgument(
            "run_name", default_value="",
            description="Kayıt klasörünün adına eklenecek etiket "
                        "(ör. run_name:=ablasyon_normal).",
        ),
        DeclareLaunchArgument(
            "note", default_value="",
            description="Koşu notu; meta.json içine yazılır "
                        "(ör. note:='küçük kutu, bandın sonunda').",
        ),

        Node(
            package="ch09_conveyor_pick_place",
            executable="perception_node",
            name="gemini_perception",
            output="screen",
            parameters=parameters,
        ),
        Node(
            package="ch09_conveyor_pick_place",
            executable="pick_place_node",
            name="gemini_pick_place",
            output="screen",
            parameters=parameters,
        ),
        # Kayıt düğümü. Görev düğümleriyle AYNI parametre dosyalarını alır;
        # `/**:` joker'i sayesinde kamera topic'leri ve mode overlay'i burada
        # da geçerli olur, yani sim/real ayrımı ikinci kez yazılmaz.
        Node(
            package="ch09_conveyor_pick_place",
            executable="recorder_node",
            name="gemini_recorder",
            output="screen",
            condition=IfCondition(LaunchConfiguration("record")),
            parameters=parameters + [{
                "record_run_name": LaunchConfiguration("run_name"),
                "record_note": LaunchConfiguration("note"),
            }],
        ),
    ])
