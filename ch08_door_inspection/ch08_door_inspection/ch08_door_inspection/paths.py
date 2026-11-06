"""
Bölüm 8 düğümlerinin ve başlatma dosyalarının varsayılan dosya yolları.

Okunan dosyalar (hazır plan, kapı ağları) kurulu paketlerden bulunur; üretilen
dosyalar (yeni plan, nokta bulutları, yörünge önbelleği) kullanıcının ev
dizinindeki tek bir çıktı klasörüne yazılır. Ara derleme ürünleri (birleştirilmiş
ağ, inanç haritası parçaları) ~/.cache/ch08_door_inspection altında tutulur.
"""

import os

from ament_index_python.packages import (get_package_share_directory,
                                         PackageNotFoundError)

PACKAGE = 'ch08_door_inspection'

#: Üretilen her şeyin kökü: ~/ros2_book_output/ch08
OUTPUT_DIR = os.path.join(os.path.expanduser('~'), 'ros2_book_output', 'ch08')


def output_path(*parts: str) -> str:
    """Çıktı klasörü altında bir yol döndürür (klasörü oluşturmaz)."""
    return os.path.join(OUTPUT_DIR, *parts)


def shipped_plans_dir() -> str:
    """Paketle gelen, kitapta kullanılan kapı planının klasörü."""
    return os.path.join(get_package_share_directory(PACKAGE), 'plans')


def door_mesh_dir() -> str:
    """
    Kapı çarpışma ağlarının (windowdoor.stl, windowlessdoor.stl) klasörünü döndürür.

    Ağlar araştırma deposundaki ur_description paketindedir; ROS 2'nin resmî
    ur_description ikili paketinde bu dosyalar YOKTUR. Paket bulunamazsa boş
    metin döner.
    """
    try:
        share = get_package_share_directory('ur_description')
    except PackageNotFoundError:
        return ''
    return os.path.join(share, 'meshes', 'ur10e', 'collision')
