"""
Toplanan bulutlardan kapıların inanç + doluluk OctoMap'ini oluşturur.

Bir düğüm grafiği değil, tek seferlik bir hattır:

    1. doors_parts_prep  (Python)  -- iki kapı STL'sini yama PCD'lerine böler; inanç
                                      haritasının temel doğrusu. Önbelleklidir; bir
                                      kaynak STL ya da yama ızgarası değişmedikçe etkisizdir.
    2. build_doors_octomap (C++)   -- bu yamalardan inanç haritasını kurar, toplanan her
                                      bulutu onun üzerine damgalar ve yama/kapı başına
                                      kaplama raporunu yazar.

Bir muayene koşusu bulutları yazdıktan SONRA çalıştırın:

    ros2 launch ch08_door_inspection doors_octomap.launch.py                 # benzetim verisi
    ros2 launch ch08_door_inspection doors_octomap.launch.py mode:=real
    ros2 launch ch08_door_inspection doors_octomap.launch.py res:=0.01 patches_y:=4
    ros2 launch ch08_door_inspection doors_octomap.launch.py mode:=real snap:=true \
        snap_radius:=2 cloud_fix:=-0.740,-5.128,0.347,0.109,0.002,0.061

GERÇEK veride snap kullanın. Oluşturucu bir vokseli yalnızca tam isabette sayar; gerçek
bulutlar 20 mm'lik voksele karşı modelden ölçülmüş 38 mm (medyan) uzakta durur. Bu
yüzden snap olmadan rapor, kameraların gerçekte topladığının çok altında okunur: 6 Ağu
2026 koşusu, kapı altlarında benzetimden DAHA ÇOK farklı nokta olmasına rağmen
benzetimin %86.2'sine karşı %38.1 aldı. Kök neden için build_doors_octomap.cpp'nin
başlığına bakın; snap bir düzeltme değil, raporlama telafisidir.

İki .ot dosyası oluşturuldukları bulutların yanına yazılır; böylece benzetim ve gerçek
haritalar birbirinin üzerine yazmaz:
    <data_dir>/beliefMap_doors_<mode>.ot
    <data_dir>/occupancyMap_doors_<mode>.ot
İncelemek için:  octovis <dosya>.ot
"""
import os
from typing import Any, List

from ament_index_python.packages import get_package_prefix
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from ch08_door_inspection import paths
from ch08_door_inspection.door_parts import (DEFAULT_DENSITY, DEFAULT_PARTS_DIR,
                                             DEFAULT_PATCHES_Y, DEFAULT_PATCHES_Z,
                                             ensure_parts)

# Kapı yürütücülerinin bulutlarını yazdığı yer (doors_inspection.launch.py'nin
# output_base_dir'i). <mode> benzetim ya da gerçek alt ağacı seçer.
DEFAULT_DATA_ROOT = paths.output_path('pcds')

# Ölçülmüş kapı hizalama hatası, "roll,pitch,yaw,x,y,z" (derece, m); yalnızca GERÇEK
# veride cloud_fix olarak verin. 7 Ağu 2026'da 12 gerçek yakalamanın tamamı (183 740
# nokta, %20 kırpılmış) üzerinde inanç ağına karşı noktadan düzleme ICP ile uyduruldu:
# kapılar world Y etrafında 5.1 derece yatık. İki kapı ayrı ayrı uydurulunca aynı açıyı
# verir (-5.04 ve -4.80) ve ayrı uydurma bu tek genel dönüşümden iyi değildir (medyan
# artık 18.5 / 17.9 ile 18.6 mm); bu yüzden tek dönüşüm kullanılır.
# Modele medyan uzaklık: 41.5 mm -> 18.6 mm.
DOORS_FIT = '-0.740,-5.128,0.347,0.109,0.002,0.061'


def _bool(context: LaunchContext, name: str) -> bool:
    return LaunchConfiguration(name).perform(context).lower() in ('true', '1', 'yes')


def _launch_setup(context: LaunchContext, *args: Any, **kwargs: Any) -> List[ExecuteProcess]:
    def cfg(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    # Önce ve senkron olarak inanç haritası girdisi: oluşturucu yama PCD'leri oluşmadan
    # başlamamalıdır; bunu ikinci bir süreç yerine burada yapmak her türlü yarış
    # olasılığını ortadan kaldırır.
    parts_dir = ensure_parts(
        parts_dir=cfg('parts_dir'),
        density=int(cfg('density')),
        patches_y=int(cfg('patches_y')),
        patches_z=int(cfg('patches_z')),
        force=_bool(context, 'force_parts'),
    )

    mode = cfg('mode')
    data_dir = cfg('data_dir') or os.path.join(DEFAULT_DATA_ROOT, f'{mode}_pcds')

    exe = os.path.join(get_package_prefix('ch08_door_inspection'), 'lib',
                       'ch08_door_inspection', 'build_doors_octomap')

    return [ExecuteProcess(
        cmd=[exe,
             '--mode', mode,
             '--parts_dir', parts_dir,
             '--data_dir', data_dir,
             # 1.0, çünkü kapı STL'leri metre cinsindendir. Şase oluşturucu milimetre CAD
             # parçaları için 1000'e böler; burada bunu vermek kapıları 1.2 mm'lik bir noktaya
             # küçültür ve kaplama raporu %0 okunurdu.
             '--parts_scale', '1.0',
             '--res', cfg('res'),
             '--z_min', cfg('z_min'),
             '--use_ur', cfg('use_ur'),
             '--use_kawasaki', cfg('use_kawasaki'),
             '--snap', cfg('snap'),
             '--snap_radius', cfg('snap_radius'),
             '--cloud_fix', cfg('cloud_fix')],
        output='screen',
    )]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            'mode', default_value='sim',
            description="sim | real -- which capture subtree to build the map from."),
        DeclareLaunchArgument(
            'data_dir', default_value='',
            description='Override the capture directory; empty means '
                        '~/ros2_book_output/ch08/pcds/<mode>_pcds.'),
        DeclareLaunchArgument(
            'parts_dir', default_value=DEFAULT_PARTS_DIR,
            description='Cache of the belief-map patch PCDs (a build artefact).'),
        DeclareLaunchArgument(
            'density', default_value=str(DEFAULT_DENSITY),
            description='Belief-map surface samples per square metre.'),
        DeclareLaunchArgument(
            'patches_y', default_value=str(DEFAULT_PATCHES_Y),
            description='Patches across each door. 1 = one part per door.'),
        DeclareLaunchArgument(
            'patches_z', default_value=str(DEFAULT_PATCHES_Z),
            description='Patches up each door. 1 = one part per door.'),
        DeclareLaunchArgument(
            'force_parts', default_value='false',
            description='Re-sample the patch PCDs even if the cache looks current.'),
        DeclareLaunchArgument(
            'res', default_value='0.02',
            description='Octomap voxel size (m). Same as the chassis job.'),
        DeclareLaunchArgument(
            'z_min', default_value='-10.0',
            description='Drop captured points below this world Z.'),
        DeclareLaunchArgument(
            'use_ur', default_value='true', description='Include the UR captures.'),
        DeclareLaunchArgument(
            'use_kawasaki', default_value='true',
            description='Include the Kawasaki captures.'),
        DeclareLaunchArgument(
            'snap', default_value='false',
            description='true -> a captured point that misses the door by up to '
                        'snap_radius voxels still counts (nearest belief voxel wins). '
                        'Rescues real-data misalignment, but flatters the coverage '
                        'number, so it is off by default and the sim numbers stay '
                        'comparable with the chassis job.'),
        DeclareLaunchArgument(
            'snap_radius', default_value='1',
            description='Snap search radius in VOXELS (metres = snap_radius * res). '
                        '1 keeps the old 26-neighbour behaviour. Use 2 for real data: '
                        'the measured door misalignment is ~8 cm at the bottom and '
                        '1-2 cm at mid height, so +-2 cm never rescued the bottom. '
                        'Measured on the 6 Aug 2026 run at res 0.02 -- radius 0/1/2/4 '
                        'gives real 38.1/48.1/55.0/58.9%, sim 86.2/87.4/87.4/87.5%: '
                        'the gain saturates at 2 and sim barely moves, so this closes '
                        'the reporting gap without inflating the sim baseline.'),
        DeclareLaunchArgument(
            'cloud_fix', default_value='0,0,0,0,0,0',
            description='Rigid correction "roll,pitch,yaw,x,y,z" (deg, m, URDF ZYX '
                        'order) applied to the CAPTURED clouds before voxelising; the '
                        'belief map is untouched so the coverage denominator stays '
                        'fixed. All zeros = off. REAL DATA ONLY -- fitted by '
                        'point-to-plane ICP on the 6 Aug 2026 run, it drops sim from '
                        '86.2% to 28.1% because sim is already aligned. '
                        'Use DOORS_FIT (see module docstring) for the measured value.'),
        OpaqueFunction(function=_launch_setup),
    ])
