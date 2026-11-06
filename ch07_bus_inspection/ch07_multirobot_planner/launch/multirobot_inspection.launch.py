"""
Bölüm 7 - 2. Aşama: çok robotlu muayenenin çevrim içi yürütülmesi.

UR10e ve Kawasaki yürütücülerini ayrı süreçlerde başlatır; varsayılan olarak
paketle gelen planı kullanır. Robot hücresinin (araştırma deposu: MoveIt 2,
denetleyiciler, Gazebo ya da gerçek robot) ayakta olması gerekir.

    $ ros2 launch ch07_multirobot_planner multirobot_inspection.launch.py
    $ ros2 launch ch07_multirobot_planner multirobot_inspection.launch.py only_ur:=true
"""

from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from ch07_multirobot_planner import paths


def generate_launch_description() -> LaunchDescription:
    """İki kolun yürütücülerini ve plan görselleştiricisini başlatır."""
    plan_file_arg = DeclareLaunchArgument(
        'plan_file', default_value=paths.shipped_plan_file(),
        description='Path to the generated multi-robot plan JSON')
    only_sim_arg = DeclareLaunchArgument(
        'only_sim', default_value='true',
        description="true -> capture ONLY the two Gazebo SICKs (ur + kawasaki). "
                    "false -> also capture the real SICKs (/sick_points, /kawasaki/pointcloud).")
    output_base_dir_arg = DeclareLaunchArgument(
        'output_base_dir', default_value=paths.output_path('pcds'),
        description='Root of the capture tree; '
                    '<base>/<sim|real>_pcds/<ur|kawasaki>_data/{pcds,poses}')
    enable_kawasaki_arg = DeclareLaunchArgument(
        'enable_kawasaki', default_value='true',
        description='Move the Kawasaki to its own coverage viewpoints and capture its camera')
    only_ur_arg = DeclareLaunchArgument(
        'only_ur', default_value='false',
        description='Real-world testing: true -> run the UR10e ONLY (Kawasaki idle).')
    only_kawasaki_arg = DeclareLaunchArgument(
        'only_kawasaki', default_value='false',
        description='Real-world testing: true -> run the Kawasaki ONLY (UR10e idle). '
                    'Both false (default) -> cooperative multi-robot run.')
    allowed_planning_time_arg = DeclareLaunchArgument(
        'allowed_planning_time', default_value='5.0',
        description='Seconds the motion planner may spend per single run')
    num_planning_attempts_arg = DeclareLaunchArgument(
        'num_planning_attempts', default_value='10',
        description='Parallel planner runs inside one plan call')
    plan_attempts_arg = DeclareLaunchArgument(
        'plan_attempts', default_value='4',
        description='How many times to re-issue the whole plan call before skipping a viewpoint')
    add_ground_plane_arg = DeclareLaunchArgument(
        'add_ground_plane', default_value='true',
        description='Add a floor collision box so MoveIt never plans an arm below ground')
    ground_plane_z_arg = DeclareLaunchArgument(
        'ground_plane_z', default_value='-0.02',
        description='World-frame height of the floor collision box top face')
    collision_padding_arg = DeclareLaunchArgument(
        'collision_padding', default_value='0.04',
        description='Safety margin (m) the arms keep from every obstacle (0.0 = off). '
                    'Applied uniformly to both arms: UR (ur10e_*) and Kawasaki (link1..link6).')
    trajectory_cache_dir_arg = DeclareLaunchArgument(
        'trajectory_cache_dir', default_value='',
        description='Where cached trajectories live. Empty -> ~/ros2_book_output/ch07/'
                    'trajectories. Give each job (chassis, doors) its own directory: '
                    'viewpoint ids such as ur_vp_000 repeat between plans.')
    use_trajectory_cache_arg = DeclareLaunchArgument(
        'use_trajectory_cache', default_value='true',
        description='true -> replay each viewpoint\'s saved trajectory from '
                    'trajectory_cache_dir if valid (deterministic, pre-validated paths); '
                    'plan+save on first run / cache miss. false -> always plan fresh with MoveIt.')
    force_replan_arg = DeclareLaunchArgument(
        'force_replan', default_value='false',
        description='true -> ignore and OVERWRITE any cached trajectory (rebuild the cache '
                    'from scratch, e.g. after changing collision padding or the SRDF).')
    wrap_goals_to_current_arg = DeclareLaunchArgument(
        'wrap_goals_to_current', default_value='true',
        description='true -> before planning, re-express every revolute goal as the '
                    '2*pi-equivalent NEAREST the arm\'s current measured angle, provided '
                    'the joint\'s URDF limits allow it. Same physical pose, but the arm '
                    'stops driving a full needless turn between viewpoints whose IK '
                    'branches differ (e.g. +100 deg then -260 deg). Joints with a range '
                    'of 2*pi or less (Kawasaki joint1/2/3/5, UR elbow) are never touched. '
                    'Set false to reproduce the old behaviour for an A/B comparison.')
    nearest_branch_ik_arg = DeclareLaunchArgument(
        'nearest_branch_ik', default_value='true',
        description='true -> before planning to a viewpoint, solve IK ourselves for that '
                    'viewpoint\'s camera pose (seeded at the arm\'s measured pose, at the '
                    'planned goal, and with the AGV rail probed either way) and go to the '
                    'CLOSEST solution that yields a path. Unwinding only removes whole '
                    'turns; this removes shoulder/elbow/wrist branch flips, which are '
                    'genuinely different configurations. The camera pose is identical in '
                    'every candidate, so captures cannot change. Set false for an A/B run.')
    branch_max_rail_shift_arg = DeclareLaunchArgument(
        'branch_max_rail_shift', default_value='0.0',
        description='Metres a branch candidate may move the AGV rail away from the '
                    'planned position. 0 (default) locks the rail exactly where the '
                    'planner put it; raise it only if you want the executor to re-resolve '
                    'the Kawasaki\'s rail redundancy at run time, which makes the AGV '
                    'drive to places the plan did not choose.')
    planner_id_arg = DeclareLaunchArgument(
        'planner_id', default_value='',
        description='OMPL planner for BOTH arms. Empty (default) -> MoveIt\'s own '
                    'fallback, RRTConnect, which is what every recorded trajectory so '
                    'far was planned with. Measured on the chassis plan 2026-09-05: no '
                    'other loadable planner beats it once num_planning_attempts is 10 '
                    '(RRTConnect 3663 deg, TRRT 4074, PRM* ~3100 but 21/24 solved, RRT '
                    'and RRT* burn 6 s a hop). Any other value must be a config name '
                    'listed for the group in ompl_planning.yaml, e.g. '
                    '"AnytimePathShorteningkConfigDefault"; an unknown name is silently '
                    'ignored by MoveIt, so check the move_group log. Re-measure with '
                    'tools/planner_bench.py before switching.')
    ik_random_seeds_arg = DeclareLaunchArgument(
        'ik_random_seeds', default_value='-1',
        description='How many RANDOM in-limits arm postures to add to the nearest-branch '
                    'IK seed list, per viewpoint. -1 (default) means "use the node\'s own '
                    'per-arm default" (12 on the Kawasaki). KDL is a LOCAL solver, so the '
                    'old seeds -- all sitting on the planned goal -- converged straight '
                    'back to it and the executor logged "1 distinct branch(es)" every '
                    'time; a /compute_ik probe with 24 random seeds returned 16-20 '
                    'DISTINCT solutions for the same poses. Only revolute joints are '
                    'randomised, and only while a trajectory is being RECORDED. '
                    '0 disables random seeds, i.e. the old behaviour.')
    ik_random_seed_arg = DeclareLaunchArgument(
        'ik_random_seed', default_value='0',
        description='RNG seed behind ik_random_seeds, so a recording run is repeatable. '
                    'Change it to draw a different set of random arm postures.')
    branch_ik_group_arg = DeclareLaunchArgument(
        'branch_ik_group', default_value='default',
        description='SRDF group the nearest-branch IK search solves in (planning is '
                    'unaffected). "default" means the node\'s own per-arm default: '
                    '"real_kawasaki_arm", the rail-free chain base_link -> link7. In the '
                    '7-DOF real_kawasaki group KDL drifts world_to_agv, and '
                    'branch_max_rail_shift then discards nearly every candidate (1 of 19 '
                    'kept the planned rail on kawa_vp_005). Empty string -> solve in the '
                    'planning group, i.e. the old behaviour ("none" does the same, and '
                    'unlike "" it survives ros2 launch\'s argument parser). '
                    'If move_group does not know '
                    'the group the node logs an error and falls back on its own.')
    kawasaki_velocity_arg = DeclareLaunchArgument(
        'kawasaki_velocity', default_value='0.007',
        description='MoveIt velocity SCALING FACTOR for the Kawasaki (fraction of the '
                    'joint_limits.yaml limits, not a speed). 0.007 gives 2.1 deg/s on '
                    'joint1-3 and 0.9 cm/s on the AGV rail (was 0.015 until 2026-09-10). '
                    'Only affects NEWLY planned paths: a cached trajectory replays at the '
                    'speed it was recorded with, so re-record (force_replan) to apply it.')
    kawasaki_acceleration_arg = DeclareLaunchArgument(
        'kawasaki_acceleration', default_value='0.007',
        description='Acceleration scaling factor. Currently a no-op: every joint in '
                    'harmony_moveit_config/config/joint_limits.yaml has '
                    'has_acceleration_limits: false, so there is no limit to scale. '
                    'Kept in step with the velocity factor for the day one is added.')
    ur_tours_arg = DeclareLaunchArgument(
        'ur_tours', default_value='2',
        description="UR10e kaç tur atsın (Kawasaki her zaman 1 tur). VARSAYILAN 2: iki kol "
                    "aynı anda başlar ama Kawasaki turunu ~7.5 dk sonra bitirir (ölçüm: UR "
                    "26 durak / 4.5 dk hareket, Kawasaki 11 durak / 8.7 dk), bu yüzden UR o "
                    "boş sürede aynı yörüngelerle ikinci turu atar. Yakalama dosyaları her "
                    "turda 1'den numaralandığı için ikinci tur birincinin bulutlarının "
                    "ÜZERİNE yazar: kapsama değişmez, bulutlar daha geç kaydedilmiş olur. "
                    "Tek tur için ur_tours:=1.")
    return_home_ur_arg = DeclareLaunchArgument(
        'return_home_ur', default_value='true',
        description='Send the UR back to its home pose after the last viewpoint.')
    return_home_kawasaki_arg = DeclareLaunchArgument(
        'return_home_kawasaki', default_value='true',
        description='Send the Kawasaki back to its home pose after the last viewpoint. '
                    'Home equals its start pose, so this also drives the AGV rail back '
                    'to where the run began. Set false to leave the arm and the AGV '
                    'wherever the last viewpoint left them.')
    branch_plan_candidates_arg = DeclareLaunchArgument(
        'branch_plan_candidates', default_value='3',
        description='How many nearest-branch candidates to actually PLAN before picking '
                    'one. Candidates are ranked by goal-space distance, which is only a '
                    'proxy: the path OMPL returns can wander well past that straight '
                    'line, so the nearest goal is not always the shortest trip. This '
                    'plans up to N of them and keeps the shortest ACTUAL trajectory. '
                    'Costs extra planning only while recording. 1 = the old '
                    'first-that-plans behaviour.')
    home_before_viewpoints_arg = DeclareLaunchArgument(
        'home_before_viewpoints', default_value="[]",
        description='Viewpoint ids that must NOT be approached directly from the previous '
                    'viewpoint. The arm detours via its start/home pose, then walks to that '
                    'viewpoint\'s RECORDED trajectory start; BOTH hops are cached like normal '
                    'trajectories, and the viewpoint keeps its own recorded path. Empty by '
                    'default: the detour existed to dodge a cable-channel pinch on the hop '
                    'into ur_vp_009, which is no longer needed now that the linear axis '
                    'tracks the trajectory properly. Set e.g. "[\'ur_vp_009\']" to re-enable.')

    # İki tek kol muayene düğümünün paylaştığı parametreler. Her düğüm kendi tam
    # parametre kümesini (varsayılanlarıyla) bildirir; bunlar başlatma dosyasından
    # değiştirilebilenlerdir.
    # only_ur / only_kawasaki / enable_kawasaki her düğümün İÇİNDE çözülür
    # (should_run): "yanlış" kolun düğümü kendiliğinden boşta kalır ve çıkar.
    common_params = {
        'plan_file': LaunchConfiguration('plan_file'),
        'only_sim': LaunchConfiguration('only_sim'),
        'output_base_dir': LaunchConfiguration('output_base_dir'),
        'enable_kawasaki': LaunchConfiguration('enable_kawasaki'),
        'only_ur': LaunchConfiguration('only_ur'),
        'only_kawasaki': LaunchConfiguration('only_kawasaki'),
        'allowed_planning_time': LaunchConfiguration('allowed_planning_time'),
        'num_planning_attempts': LaunchConfiguration('num_planning_attempts'),
        'plan_attempts': LaunchConfiguration('plan_attempts'),
        'add_ground_plane': LaunchConfiguration('add_ground_plane'),
        'ground_plane_z': LaunchConfiguration('ground_plane_z'),
        'collision_padding': LaunchConfiguration('collision_padding'),
        'use_trajectory_cache': LaunchConfiguration('use_trajectory_cache'),
        'trajectory_cache_dir': LaunchConfiguration('trajectory_cache_dir'),
        'force_replan': LaunchConfiguration('force_replan'),
        'wrap_goals_to_current': LaunchConfiguration('wrap_goals_to_current'),
        'nearest_branch_ik': LaunchConfiguration('nearest_branch_ik'),
        'branch_max_rail_shift': LaunchConfiguration('branch_max_rail_shift'),
        'branch_plan_candidates': LaunchConfiguration('branch_plan_candidates'),
        'planner_id': LaunchConfiguration('planner_id'),
        'kawasaki_velocity': LaunchConfiguration('kawasaki_velocity'),
        'kawasaki_acceleration': LaunchConfiguration('kawasaki_acceleration'),
        'return_home_ur': LaunchConfiguration('return_home_ur'),
        'return_home_kawasaki': LaunchConfiguration('return_home_kawasaki'),
        'home_before_viewpoints': ParameterValue(
            LaunchConfiguration('home_before_viewpoints'), value_type=List[str]),
    }

    # Birbirinden bağımsız iki tek kol düğümü (ayrı süreçler -> yapıları gereği paralel).
    ur_node = Node(
        package='ch07_multirobot_planner',
        executable='ur_inspection_node',
        name='ur_inspection_node',
        output='screen',
        # tours YALNIZ UR'ye verilir; Kawasaki tek tur atmaya devam eder (zaten geç
        # biten kol odur). value_type=int şart: LaunchConfiguration metin döndürür ve
        # parametre int olarak bildirilmiştir.
        parameters=[common_params, {
            'tours': ParameterValue(LaunchConfiguration('ur_tours'), value_type=int),
        }],
    )
    # Dal IK ayarları YALNIZCA KAWASAKI'ye gider. Onun KDL çözücüsü ve fazladan eksenli
    # AGV rayı için vardırlar; bunları UR'ye vermek (`mode: global` ile zaten rastgele
    # yeniden başlayan pick_ik) ölçülmüş ve çalışan bir yapıyı değiştirirdi.
    kawasaki_node = Node(
        package='ch07_multirobot_planner',
        executable='kawasaki_inspection_node',
        name='kawasaki_inspection_node',
        output='screen',
        parameters=[common_params, {
            'ik_random_seeds': LaunchConfiguration('ik_random_seeds'),
            'ik_random_seed': LaunchConfiguration('ik_random_seed'),
            'branch_ik_group': LaunchConfiguration('branch_ik_group'),
        }],
    )

    # Yanında RViz ara nokta görselleştiricisi: aynı plan JSON'undan İKİ kolun bakış
    # noktası oklarını çizer (UR yeşil->kırmızı, Kawasaki koyu->açık mavi).
    visualizer_node = Node(
        package='ch07_multirobot_planner',
        executable='multirobot_viewpoint_visualizer',
        name='multirobot_viewpoint_visualizer',
        output='screen',
        parameters=[{
            'plan_file': LaunchConfiguration('plan_file'),
        }],
    )

    return LaunchDescription([
        plan_file_arg, only_sim_arg, output_base_dir_arg, enable_kawasaki_arg,
        only_ur_arg, only_kawasaki_arg,
        allowed_planning_time_arg, num_planning_attempts_arg, plan_attempts_arg,
        add_ground_plane_arg, ground_plane_z_arg, collision_padding_arg,
        trajectory_cache_dir_arg, use_trajectory_cache_arg, force_replan_arg,
        wrap_goals_to_current_arg,
        nearest_branch_ik_arg, branch_max_rail_shift_arg, branch_plan_candidates_arg,
        ik_random_seeds_arg, ik_random_seed_arg, branch_ik_group_arg, planner_id_arg,
        kawasaki_velocity_arg, kawasaki_acceleration_arg,
        ur_tours_arg, return_home_ur_arg, return_home_kawasaki_arg,
        home_before_viewpoints_arg,
        ur_node, kawasaki_node, visualizer_node,
    ])
