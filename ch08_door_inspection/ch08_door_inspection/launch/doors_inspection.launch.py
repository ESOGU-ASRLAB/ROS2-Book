"""
Kapı planını iki kolda yürütür.

Bu dosya yürütücüleri yeniden yazmaz. ch07_multirobot_planner'ın muayene başlatmasını
kapı planı ve kapıya ait bir çıktı ağacıyla dâhil eder; böylece UR ve Kawasaki
düğümleri, yörünge önbelleği, 2*pi sarma ve en yakın dal IK'sı şase işinde zaten
doğrulanmış olanlarla aynıdır ve orada yapılan her sonraki düzeltme buraya da yansır.

Çıktılar şase işininkilerden ayrı tutulur:
  * yörünge önbelleği -> ~/ros2_book_output/ch08/trajectories/
  * toplanan bulutlar -> <output_base_dir>/{sim,real}_pcds/{ur,kawasaki}_data/
    (varsayılan output_base_dir: ~/ros2_book_output/ch08/pcds)

Tipik kullanım:
    ros2 launch ch08_door_inspection doors_inspection.launch.py
    ros2 launch ch08_door_inspection doors_inspection.launch.py only_sim:=false
    ros2 launch ch08_door_inspection doors_inspection.launch.py force_replan:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from ch08_door_inspection import paths

# Kitapta kullanılan kapı planı paketle birlikte gelir
PLANS_DIR = paths.shipped_plans_dir()

# Yukarı akıştaki muayene başlatmasına olduğu gibi iletilen argümanlar. Liste olarak
# tutulur; orada bir ayar eklemek burada yeni bir blok değil tek bir satır gerektirir.
FORWARDED = [
    ('only_sim', 'true',
     'true -> capture only the two Gazebo SICKs; false -> also the real ones.'),
    ('enable_kawasaki', 'true', 'Run the Kawasaki as well as the UR.'),
    ('only_ur', 'false', 'true -> UR only, Kawasaki idle.'),
    ('only_kawasaki', 'false', 'true -> Kawasaki only, UR idle.'),
    ('allowed_planning_time', '5.0', 'Seconds per planning run.'),
    ('num_planning_attempts', '10', 'Parallel planner runs inside one plan call.'),
    ('plan_attempts', '4', 'Re-issues of the whole plan call before skipping a viewpoint.'),
    ('add_ground_plane', 'true', 'Add a floor collision box.'),
    ('ground_plane_z', '-0.02', 'World Z of the floor box top face.'),
    ('collision_padding', '0.04', 'Safety margin (m) both arms keep from every obstacle.'),
    ('use_trajectory_cache', 'true', 'Replay saved trajectories when they are still valid.'),
    # Kapı işinin kendi önbellek klasörü: şase planıyla aynı bakış noktası kimlikleri
    # (ur_vp_000, ...) kullanıldığı için ortak bir klasör iki işin yörüngelerini karıştırırdı.
    ('trajectory_cache_dir', paths.output_path('trajectories'),
     'Doors trajectory cache, kept apart from the chassis cache.'),
    ('force_replan', 'false', 'Ignore and overwrite the cache (use after a scene change).'),
    ('wrap_goals_to_current', 'true', 'Re-express goals as the nearest 2*pi equivalent.'),
    ('nearest_branch_ik', 'true', 'Go to the closest IK branch of the same camera pose.'),
    ('branch_max_rail_shift', '0.0', 'Metres a branch may move the AGV rail (0 locks it).'),
    ('branch_plan_candidates', '3',
     'Plan this many branch candidates and keep the shortest ACTUAL path (1 = the old '
     'first-that-plans behaviour).'),
    # KAPIYA ÖZGÜ: şase işinin kullandığı 0.015 yerine 0.03, yani iki kat hız; joint1-3'te
    # 9 derece/sn ve AGV rayında 3.7 cm/sn. Doğrulanmış şase işine dokunulmasın diye
    # düğümde değil burada geçersiz kılınır.
    ('kawasaki_velocity', '0.03',
     'MoveIt velocity scaling factor for the Kawasaki (fraction of its joint limits).'),
    ('kawasaki_acceleration', '0.03',
     'Acceleration scaling factor. A no-op today: joint_limits.yaml declares no '
     'acceleration limits, so there is nothing to scale. Kept in step with the '
     'velocity factor.'),
    ('return_home_ur', 'true', 'Send the UR home after the last viewpoint.'),
    ('return_home_kawasaki', 'true',
     'Send the Kawasaki home after the last viewpoint. Home is its start pose, so the '
     'AGV rail drives back too.'),
    # Yine boş: gerçek koşu doğrudan hamlenin donanımda sorunsuz olduğunu gösterince
    # kawa_vp_002'nin eve uğraması kaldırıldı. Ölçülebilir hiçbir yararı olmadan ~1.5 dk
    # (2 m fazladan ray yolu ve 13 rad fazladan eklem yolu) harcatıyordu. Geri getirmek için
    # "['kawa_vp_002']" verin ya da öncülünden doğrudan değil evden yaklaşılması
    # gereken başka bir bakış noktasının adını yazın.
    ('home_before_viewpoints', "[]",
     'Viewpoint ids approached via the home pose instead of directly from the previous '
     'viewpoint.'),
]


def generate_launch_description() -> LaunchDescription:
    upstream = os.path.join(
        get_package_share_directory('ch07_multirobot_planner'),
        'launch', 'multirobot_inspection.launch.py')

    args = [
        DeclareLaunchArgument(
            'plan_file',
            default_value=os.path.join(PLANS_DIR, 'doors_viewpoint_plan.json'),
            description='Doors plan produced by doors_planning.launch.py.'),
        DeclareLaunchArgument(
            'output_base_dir',
            default_value=paths.output_path('pcds'),
            description='Root of the doors capture tree; kept apart from the chassis '
                        'captures so the two jobs never overwrite each other.'),
    ]
    args += [DeclareLaunchArgument(n, default_value=d, description=h)
             for n, d, h in FORWARDED]

    include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(upstream),
        launch_arguments=[('plan_file', LaunchConfiguration('plan_file')),
                          ('output_base_dir', LaunchConfiguration('output_base_dir'))]
        + [(n, LaunchConfiguration(n)) for n, _, _ in FORWARDED],
    )

    return LaunchDescription(args + [include])
