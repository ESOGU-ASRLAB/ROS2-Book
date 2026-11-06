"""
Bölüm 7 düğümlerinin varsayılan dosya yolları.

Okunan dosyalar (hazır plan, şase ağı) kurulu paketlerden bulunur; üretilen
dosyalar (yeni plan, nokta bulutları, yörünge önbelleği, şekiller) kullanıcının
ev dizinindeki tek bir çıktı klasörüne yazılır. Böylece kod hiçbir makineye özgü
mutlak yol içermez ve kurulum klasörüne (--symlink-install ile kaynak ağacına)
yazmaz.
"""

import os

from ament_index_python.packages import (get_package_share_directory,
                                         PackageNotFoundError)

PACKAGE = 'ch07_multirobot_planner'

#: Üretilen her şeyin kökü: ~/ros2_book_output/ch07
OUTPUT_DIR = os.path.join(os.path.expanduser('~'), 'ros2_book_output', 'ch07')


def output_path(*parts: str) -> str:
    """Çıktı klasörü altında bir yol döndürür (klasörü oluşturmaz)."""
    return os.path.join(OUTPUT_DIR, *parts)


def shipped_plan_file() -> str:
    """Paketle gelen, kitapta kullanılan çok robotlu plan dosyası."""
    return os.path.join(get_package_share_directory(PACKAGE), 'plans',
                        'multirobot_viewpoint_plan.json')


def chassis_mesh_file() -> str:
    """
    Otobüs şasesi çarpışma ağının yolunu döndürür.

    Ağ, araştırma deposundaki ur_description paketindedir; ROS 2'nin resmî
    ur_description ikili paketinde bu dosya YOKTUR. Araştırma deposunun çalışma
    alanı kaynaklanmamışsa (source) dönen yol resmî pakete işaret eder ve
    planlayıcı "Mesh path invalid" hatası verir.

    Paket kurulu değilse boş metin döner; planlayıcı bu durumda mesh_path
    parametresinin elle verilmesini ister.
    """
    try:
        share = get_package_share_directory('ur_description')
    except PackageNotFoundError:
        return ''
    return os.path.join(share, 'meshes', 'ur10e', 'collision', 'chassis.stl')
