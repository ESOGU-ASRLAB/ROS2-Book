#!/usr/bin/env python3
"""
Vakumlu kavrama pozu: ER 2'nin 2D noktasından emme noktası + yaklaşma yönü.

NEDEN: ER 2 tek bir 2D nokta verir, oryantasyon vermez. Parmaklı bir gripper'da
sabit "yukarıdan aşağı" yaklaşma çoğu zaman idare eder. Vakumlu kapta ETMEZ:
emme kabı yüzeye DİK oturmak zorunda, yoksa sızdırır ve parça hiç tutulmaz.
Yüzey normali bu yüzden opsiyonel bir iyileştirme değil, işin gereği.

Yaklaşım: ER 2'nin gösterdiği pikselin çevresindeki bulut noktalarına PCA ile
düzlem oturtulur. En küçük özvektör = yüzey normali, en küçük özdeğerin karekökü
= düzlemsellik artığı. Artık büyükse (kenar, köşe, delik) kavrama REDDEDİLİR -
vakum orada zaten tutmaz, denemek boşa hareket.

suction_cup frame'i tool0 ile eş hizalı (vacuum_gripper_joint ve
suction_cup_joint ikisi de rpy=0 0 0), dolayısıyla emme ekseni suction_cup'ın
+Z'sidir. Kavrama kuaterniyonu tool +Z'yi yüzeyin İÇİNE (-normal) bakacak
şekilde kurar.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from sensor_msgs.msg import PointCloud2


Vec3 = Tuple[float, float, float]


@dataclass
class SurfacePatch:
    """ER 2'nin gösterdiği noktanın çevresine oturtulan yerel düzlem."""

    centroid: Vec3  # yama merkezi (bulutun frame'inde)
    normal: Vec3  # birim normal, kameraya doğru yönlendirilmiş
    rms_residual: float  # düzlemsellik artığı (m); küçük = düz
    inliers: int  # BANTTA kalan, düzlem oturtmada kullanılan nokta
    extent: float  # yamanın yaklaşık yarıçapı (m)
    # Banda GİREN toplam nokta. inliers ile birlikte bandın ne kadarını
    # eledigini verir; bu oran olmadan "bant çalıştı mı" sorusu ölçülemez,
    # yalnız normalin sonucundan dolaylı olarak tahmin edilebilirdi.
    patch_points: int = 0


def _field_offsets(cloud: PointCloud2) -> Optional[Tuple[int, int, int]]:
    offsets = {field.name: field.offset for field in cloud.fields}
    if not {"x", "y", "z"} <= offsets.keys():
        return None
    return (offsets["x"], offsets["y"], offsets["z"])


def gather_patch_points(
    cloud: PointCloud2, u: int, v: int, radius_px: int
) -> Optional[np.ndarray]:
    """(u, v) çevresindeki geçerli bulut noktalarını Nx3 dizi olarak toplar."""
    if cloud.height <= 1:
        return None
    offsets = _field_offsets(cloud)
    if offsets is None:
        return None
    x_off, y_off, z_off = offsets

    points: List[Tuple[float, float, float]] = []
    for dv in range(-radius_px, radius_px + 1):
        vv = v + dv
        if not (0 <= vv < cloud.height):
            continue
        for du in range(-radius_px, radius_px + 1):
            uu = u + du
            if not (0 <= uu < cloud.width):
                continue
            if du * du + dv * dv > radius_px * radius_px:
                continue  # kare değil, daire
            base = (vv * cloud.width + uu) * cloud.point_step
            try:
                x = struct.unpack_from("f", cloud.data, base + x_off)[0]
                y = struct.unpack_from("f", cloud.data, base + y_off)[0]
                z = struct.unpack_from("f", cloud.data, base + z_off)[0]
            except struct.error:
                continue
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            points.append((x, y, z))

    if not points:
        return None
    return np.asarray(points, dtype=np.float64)


def fit_surface(
    points: np.ndarray,
    depth_band: float = 0.05,
    min_points: int = 25,
    anchor: Optional[Sequence[float]] = None,
) -> Optional[SurfacePatch]:
    """
    Nokta bulutuna PCA ile düzlem oturtur.

    depth_band: çapa derinliğinden bu kadar uzaktaki noktalar atılır. ER 2
    nesnenin kenarına yakın işaret ettiğinde yamaya arka plan karışıyor; bunu
    ayıklamazsak normal, nesne ile arka plan arasında bir yere bakar ve emme
    kabı yüzeye eğik oturur.

    anchor: ER 2'nin işaret ettiği pikselin 3B noktası. Çapa olarak bunu
    kullanmak şart: yamanın MEDYANI kullanılırsa, yama iki yüzeye yaklaşık eşit
    bölündüğünde (yani tam bandın var olma sebebi olan kenar durumunda) medyan
    iki yüzeyin arasındaki boşluğa düşer ve her iki yüzey birden elenir.
    None verilirse yamanın medyan noktasına düşülür.

    "Derinlik" BAKIŞ IŞINI boyunca ölçülür, bir koordinat ekseni boyunca değil.
    Eskiden z bileşeni kullanılıyordu; optik frame'de (z = ileri) doğru, ama
    Gazebo bulutu x-ileri konvansiyonunda geliyor (gemini_params.yaml'daki
    deprojection notu) - orada z YANAL bir eksen ve band yanlış yönde kesiyor,
    yani arka planı hiç elemiyordu. Işına izdüşüm iki konvansiyonda da aynı
    şeyi ölçer, çünkü kamera her ikisinde de orijinde.
    """
    if points is None or len(points) < min_points:
        return None

    if anchor is not None:
        anchor_point = np.asarray(anchor, dtype=np.float64)
    else:
        anchor_point = np.median(points, axis=0)

    ray_length = float(np.linalg.norm(anchor_point))
    if ray_length < 1e-6:
        return None
    ray = anchor_point / ray_length

    band = points[np.abs(points @ ray - ray_length) <= depth_band]
    if len(band) < min_points:
        return None

    centroid = band.mean(axis=0)
    centered = band - centroid

    # Kovaryansın en küçük özvektörü = düzlem normali
    try:
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vh[2, :]

    # Artık: noktaların düzleme ortalama KARESEL uzaklığı.
    #
    # DÜZELTİLDİ 17 Ağu 2026: burada `np.sqrt(d ** 2).mean()` yazıyordu, yani
    # karekök ortalamadan ÖNCE alınıyordu - bu RMS değil, mutlak sapmanın
    # ortalaması (MAD). MAD her zaman RMS'ten küçüktür, yani max_surface_rms
    # kapısı yazılı olduğundan DAHA GEVŞEK çalışıyordu. Fark, tam da kapının
    # yakalaması gereken durumda büyük: yamaya arka plan karıştığında dağılım
    # iki tepeli olur ve MAD, RMS'in çok altında kalır.
    #
    # ÖLÇÜLDÜ (bant üstünde 29 mm'lik kutu, 12 px yama, kenara yakın işaret):
    #   kirlenme | MAD (eski) |  RMS  | düzlem eğimi | 15 cm'de yanal kayma
    #   ---------|------------|-------|--------------|---------------------
    #      4%    |   2.9 mm   | 5.3mm |    8.0°      |      21 mm
    #      6%    |   4.0 mm   | 6.2mm |   10.9°      |      29 mm
    # Yani %4-6 kirlenmede eski kapı GEÇİRİYOR, doğru kapı REDDEDİYOR - ve
    # geçen yamanın normali on derece eğik. Eğik normal iki belirti üretir:
    # yaklaşma pozu nesnenin yanına düşer, iniş nesneyi süpürür.
    residual = centered @ normal
    rms = float(np.sqrt((residual ** 2).mean()))
    extent = float(np.linalg.norm(centered, axis=1).max())

    # Normali kameraya doğru çevir. Optik frame'de kamera orijinde ve +Z ileri
    # bakar, yani tüm noktaların z'si pozitif. Kameraya bakan normal için
    # normal . centroid < 0 olmalı.
    if float(normal @ centroid) > 0.0:
        normal = -normal

    norm_len = float(np.linalg.norm(normal))
    if norm_len < 1e-9:
        return None
    normal = normal / norm_len

    return SurfacePatch(
        centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        rms_residual=rms,
        inliers=int(len(band)),
        extent=extent,
        patch_points=int(len(points)),
    )


def _matrix_to_quaternion(matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """
    3x3 rotasyon matrisini (x, y, z, w) kuaterniyonuna çevirir.

    Shepperd yöntemi: en büyük köşegen terimden türetilir, böylece sıfıra
    bölme ve küçük w'de duyarlılık kaybı olmaz.
    """
    m = matrix
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    quat = np.array([x, y, z, w], dtype=np.float64)
    return tuple(float(value) for value in quat / np.linalg.norm(quat))


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def quaternion_to_matrix(quat_xyzw: Sequence[float]) -> np.ndarray:
    """Bir xyzw kuaterniyonunu 3x3 rotasyon matrisine çevirir."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def snap_normal_to_axis(
    normal_world: Sequence[float],
    tolerance_deg: float,
) -> Tuple[Tuple[float, float, float], float]:
    """
    Normali en yakın dünya eksenine oturtur (tolerans içindeyse).

    (snapped_normal, sapma_derecesi) döndürür; tolerans dışındaysa normal
    olduğu gibi geri gelir.

    Neden: bant ve raf yüzeyleri işlenmiş, dünya eksenlerine göre düz
    yüzeyler. Normali derinlik yamasından PCA ile kestirince birkaç derecelik
    gürültü biniyor - ölçüldü: bırakma pozunda kap dikeyden 3.99 derece
    sapmıştı. Ø23 mm'lik emme kabında 4 derece, bir kenarda ~1.6 mm boşluk
    demek ve gerçek donanımda vakum sızdırır.

    tolerance_deg = 0 ile kapatılır; yüzeyin GERÇEKTEN eğik olabileceği
    işlerde kapatılmalı, çünkü bu bir modelleme varsayımıdır.
    """
    vector = np.asarray(normal_world, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        raise ValueError("Yüzey normali sıfır uzunlukta")
    vector = vector / length

    axes = (
        np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]),
    )
    dots = [float(vector @ axis) for axis in axes]
    best = int(np.argmax(dots))
    deviation = math.degrees(math.acos(max(-1.0, min(1.0, dots[best]))))

    if tolerance_deg > 0.0 and deviation <= tolerance_deg:
        snapped = axes[best]
        return (float(snapped[0]), float(snapped[1]), float(snapped[2])), deviation
    return (float(vector[0]), float(vector[1]), float(vector[2])), deviation


def _orthonormal_basis(primary: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Primary vektörünü ilk sütun yapan sağ el ortonormal tabanı döndürür (3x3, det=+1)."""
    ref = np.asarray(reference, dtype=np.float64)
    if abs(float(ref @ primary)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
        if abs(float(ref @ primary)) > 0.95:
            ref = np.array([0.0, 0.0, 1.0])
    second = np.cross(ref, primary)
    length = np.linalg.norm(second)
    if length < 1e-9:
        raise ValueError("Referans vektör birincil eksene paralel")
    second /= length
    third = np.cross(primary, second)
    return np.column_stack((primary, second, third))


def matrix_to_quaternion(matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """
    3x3 rotasyon matrisini (x, y, z, w) kuaterniyonuna çevirir.

    Ölçülen eksen üçlüsünü MoveIt'in beklediği kuaterniyona çevirmek için
    kullanılır (bkz. payload ölçümü).
    """
    return _matrix_to_quaternion(np.asarray(matrix, dtype=np.float64))


def surface_frame_quaternion(
    normal_world: Sequence[float], reference: Sequence[float] = (1.0, 0.0, 0.0)
) -> Tuple[float, float, float, float]:
    """
    +Z'si yüzey normali boyunca olan bir frame'in kuaterniyonu.

    Kavranan parçanın çarpışma kutusunu MoveIt sahnesine yerleştirmek için:
    kutunun yüksekliği yüzeye dik olmalı. KAVRAMA kuaterniyonu bu iş için
    yanlıştır - o, uç elemanın kendi eğik eksenini (bu hücrede 30 derece)
    içerir ve kutuyu eğik gösterir.

    Yüzey içindeki dönme ÖLÇÜLMÜYOR: ER bir nokta veriyor, parçanın yönünü
    değil. `reference` ile dünya eksenine sabitleniyor; dikdörtgen bir parçada
    bu, gerçek yönelimden 90 dereceye kadar sapabilir. Kutunun kenar
    uzunlukları farklıysa çarpışma gövdesi o kadar yanlış olur - kritikse
    parçanın yönelimi ayrıca kestirilmeli.
    """
    normal = np.asarray(normal_world, dtype=np.float64)
    length = float(np.linalg.norm(normal))
    if length < 1e-9:
        raise ValueError("Yüzey normali sıfır uzunlukta")
    normal = normal / length
    # _orthonormal_basis primary'yi İLK sütuna koyar; bize +Z lazım, o yüzden
    # sütunları kaydırıyoruz: (n, a, b) -> (a, b, n)
    basis = _orthonormal_basis(normal, np.asarray(reference, dtype=np.float64))
    rotation = np.column_stack((basis[:, 1], basis[:, 2], basis[:, 0]))
    return _matrix_to_quaternion(rotation)


def approach_vector_from(spec: Union[str, Sequence[float]]) -> np.ndarray:
    """'x'/'y'/'z' ya da üç bileşenli vektörü birim yaklaşma vektörüne çevirir."""
    if isinstance(spec, str):
        key = spec.lower()
        if key not in AXIS_INDEX:
            raise ValueError(f"approach ekseni 'x', 'y' veya 'z' olmalı, {spec!r} geldi")
        vector = np.zeros(3)
        vector[AXIS_INDEX[key]] = 1.0
        return vector
    vector = np.asarray(spec, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"approach vektörü 3 bileşenli olmalı, {spec!r} geldi")
    length = np.linalg.norm(vector)
    if length < 1e-9:
        raise ValueError("approach vektörü sıfır uzunlukta")
    return vector / length


def approach_quaternion(
    surface_normal_world: Sequence[float],
    approach_vector: Sequence[float] = (0.0, 0.0, 1.0),
    yaw_reference: Sequence[float] = (1.0, 0.0, 0.0),
) -> Tuple[float, float, float, float]:
    """
    Emme eksenini yüzeyin içine bakacak şekilde hizalayan kuaterniyon.

    approach_vector, uç eleman frame'inde emme yönüdür ve HERHANGİ bir vektör
    olabilir - eksene hizalı olmak zorunda değil. Bu hücrede zorunlu:
    `ur10e_suction_cup` mesh'ine dönel yüzey ekseni uyduruldu ve kap, frame'in
    X ekseninden **30.00 derece** eğik çıktı, yani (cos30, sin30, 0). Önce
    "+Z" varsayılmıştı (kap yatay geliyordu), sonra "+X" (30 derece eğik
    kalıyordu).

    Emme kabı dönel simetrik olduğu için yaw serbesttir; yaw_reference sadece
    tekrarlanabilir ve IK için makul bir çözüm seçmeye yarar.
    """
    normal = np.asarray(surface_normal_world, dtype=np.float64)
    length = np.linalg.norm(normal)
    if length < 1e-9:
        raise ValueError("Yüzey normali sıfır uzunlukta")
    normal = normal / length

    tool_axis = approach_vector_from(approach_vector)
    # Yaklaşma yönü: yüzeyin İÇİNE, yani dışa bakan normalin tersi.
    world_axis = -normal

    # R = W * T^T, iki sağ el tabanı arasındaki dönme. R @ tool_axis = world_axis
    # olur, çünkü T^T tool_axis'i (1,0,0)'a, W de onu world_axis'e taşır.
    tool_basis = _orthonormal_basis(tool_axis, np.array([0.0, 0.0, 1.0]))
    world_basis = _orthonormal_basis(world_axis, np.asarray(yaw_reference, dtype=np.float64))
    rotation = world_basis @ tool_basis.T
    return _matrix_to_quaternion(rotation)


def is_graspable(
    patch: SurfacePatch,
    max_rms: float,
    min_inliers: int,
    max_tilt_deg: float,
    up_axis_world: Sequence[float] = (0.0, 0.0, 1.0),
    normal_world: Optional[Sequence[float]] = None,
) -> Tuple[bool, str]:
    """
    Yamanın vakumla tutulabilir olup olmadığını denetler.

    Döndürdüğü gerekçe log'a yazılır; "neden kavramadı" sorusunun cevabı
    doğrudan burada görünsün diye metin olarak veriliyor.
    """
    if patch.inliers < min_inliers:
        return False, f"yetersiz nokta ({patch.inliers} < {min_inliers})"
    if patch.rms_residual > max_rms:
        return False, (
            f"yüzey düz değil (artık {patch.rms_residual * 1000:.1f} mm > "
            f"{max_rms * 1000:.1f} mm) - kenar veya köşe olabilir"
        )
    if normal_world is not None and max_tilt_deg < 180.0:
        normal = np.asarray(normal_world, dtype=np.float64)
        normal /= max(1e-9, np.linalg.norm(normal))
        up = np.asarray(up_axis_world, dtype=np.float64)
        up /= max(1e-9, np.linalg.norm(up))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(normal @ up)))))
        if tilt > max_tilt_deg:
            return False, f"yüzey fazla eğik ({tilt:.0f}° > {max_tilt_deg:.0f}°)"
    return True, "uygun"
