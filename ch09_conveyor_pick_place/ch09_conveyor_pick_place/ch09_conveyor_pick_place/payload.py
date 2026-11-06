#!/usr/bin/env python3
"""
Kavranacak parçanın boyutunu DERİNLİKTEN ölçer (yönlü sınırlayıcı kutu).

NEDEN: kavranan parça artık robotun bir parçasıdır ve MoveIt'in onu görmesi
gerekir. Boyutu elle vermek tek bir parçaya kilitler; farklı parçalarla
çalışılacaksa her seferinde config düzenlemek gerekir - ve unutulduğunda
belirti sessizdir (kol parçayı bir yere sürter).

ÖLÇÜM FİKRİ. Derinlik kamerası mesafe ölçer, dolayısıyla YÜKSEKLİK doğrudan
ölçülen büyüklüktür; zor olan yanal boyutlardır, çünkü "cisim nerede bitiyor"
kararı bir segmentasyon problemidir. Her ikisi de şu üç adımda çıkar:

  1. Destek düzlemi: parçanın üst yüzü zaten uydurulmuş durumda (fit_surface).
     Parça düz bir yüzeyin üstünde durduğuna göre destek düzlemi ona
     PARALELDİR; tek bilinmeyen uzaklığı. Noktaların normal boyunca izdüşümü
     1B bir histograma dönüşür ve destek, üst yüzün altındaki baskın tepedir.
  2. Segmentasyon: destek düzleminin üstünde kalan ve ER'nin gösterdiği piksele
     BAĞLI olan pikseller. Bağlılık şart - aynı yükseklikteki komşu bir cisim
     yoksa kutuya karışır.
  3. Kutu: o noktaların düzlem içindeki 2B PCA'sı uzunluk, genişlik ve düzlem
     içi YÖNELİMİ verir; yükseklik ise destek düzleminden en yüksek noktadır.

NE ÖLÇÜLEMEZ (hepsi sessizce yanlış sonuç verir, bilerek kullanın):
  * Destek düzleminin ALTI. Ayaklı, oyuk tabanlı ya da çıkıntılı parçada
    ölçülen şey siluettir, gövde değil.
  * "Düzlemin üstünde duruyor" varsayımı. Parça başka bir parçanın üstündeyse
    ya da bir gözün içindeyse referans düzlem yanlıştır.
  * Görünmeyen yüzler. Eğik bakışta arka taraf oklüzyonda kalır; ayak izi
    EKSİK çıkar (yükseklik etkilenmez).
  * Malzeme. ToF; mat siyah, parlak metal ve şeffaf yüzeylerde ya veri
    vermez ya da saçma verir. Yeterli piksel yoksa ölçüm None döner.

Bu yüzden sonuç HER ZAMAN bir payla şişirilir ve başarısızlıkta çağıran
tarafın yapılandırılmış yedeğe düşmesi beklenir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
    CV_AVAILABLE = True
except ImportError:  # pragma: no cover
    CV_AVAILABLE = False

Vec3 = Tuple[float, float, float]


@dataclass
class PayloadBox:
    """Ölçülen parça kutusu, bulutun kendi frame'inde."""

    size: Vec3  # (uzunluk, genişlik, yükseklik) m - şişirilmiş
    size_measured: Vec3  # pay eklenmeden ölçülen
    centre: Vec3  # kutunun merkezi
    axes: np.ndarray  # 3x3, sütunlar: uzun kenar, kısa kenar, normal
    pixels: int  # segmentte kaç piksel vardı
    support_distance: float  # üst yüzden destek düzlemine mesafe (= yükseklik)


def organized_xyz(cloud: Any) -> Optional[np.ndarray]:
    """
    PointCloud2'yi (H, W, 3) float dizisine çevirir (vektörize).

    gather_patch_points struct ile tek tek okuyor; bu iş için tüm kare
    gerektiğinden o yol çok yavaş kalıyor. Alan ofsetleri tek tek alınıyor:
    x/y/z'nin bitişik olduğunu VARSAYMAK bazı sürücülerde yanlış.
    """
    if cloud.height <= 1:
        return None
    offsets = {f.name: f.offset for f in cloud.fields}
    if not {"x", "y", "z"} <= offsets.keys():
        return None
    raw = np.frombuffer(cloud.data, dtype=np.uint8)
    expected = cloud.height * cloud.width * cloud.point_step
    if raw.size < expected:
        return None
    raw = raw[:expected].reshape(cloud.height, cloud.width, cloud.point_step)

    def field(offset: int) -> np.ndarray:
        chunk = raw[:, :, offset:offset + 4]
        return np.ascontiguousarray(chunk).view(np.float32).reshape(
            cloud.height, cloud.width)

    return np.dstack([field(offsets[k]) for k in ("x", "y", "z")]).astype(np.float64)


def _histogram_mode(values: np.ndarray, bin_width: float) -> Optional[float]:
    """
    Histogramın en kalabalık kovasının merkezini döndürür.

    Medyan DEĞİL: destek düzlemi çoğunlukta olsa bile medyan, cismin
    pikselleriyle karışıp aradaki boşluğa düşebilir.
    """
    if values.size < 20:
        return None
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < bin_width:
        return 0.5 * (lo + hi)
    bins = max(4, int(np.ceil((hi - lo) / bin_width)))
    counts, edges = np.histogram(values, bins=bins)
    index = int(counts.argmax())
    return 0.5 * float(edges[index] + edges[index + 1])


def find_support_plane(
    heights: np.ndarray, min_thickness: float = 0.003, bin_width: float = 0.002
) -> Optional[float]:
    """
    Üst yüze göre destek düzleminin yüksekliğini (negatif) döndürür.

    `heights`: noktaların üst yüz düzlemine göre işaretli yüksekliği. Üst yüz
    0'da, altındaki her şey negatif. Destek, min_thickness'ten daha aşağıdaki
    baskın tepedir.
    """
    below = heights[heights < -min_thickness]
    if below.size < 20:
        return None
    return _histogram_mode(below, bin_width)


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Verilen normale dik iki birim vektörü döndürür."""
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(seed @ normal)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    first = np.cross(normal, seed)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    return first, second


def _connected_mask(mask: np.ndarray, seed_uv: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Tohum (seed) pikseline BAĞLI bileşeni döndürür.

    Bağlılık olmadan, aynı yükseklikteki komşu bir cisim (yan yana duran ikinci
    bir kutu, rafın kenarı) tek bir dev kutuya birleşir.
    """
    u, v = seed_uv
    if not (0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]):
        return None
    if not mask[v, u]:
        # ER'nin pikseli tam kenara denk gelmiş olabilir: küçük bir komşulukta ara.
        window = mask[max(0, v - 3):v + 4, max(0, u - 3):u + 4]
        if not window.any():
            return None
        local = np.argwhere(window)[0]
        v, u = max(0, v - 3) + int(local[0]), max(0, u - 3) + int(local[1])

    if CV_AVAILABLE:
        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        if count <= 1:
            return None
        return labels == labels[v, u]

    # cv2 yoksa basit taşma doldurma (yığın tabanlı, özyinelemesiz)
    out = np.zeros_like(mask)
    stack = [(v, u)]
    height, width = mask.shape
    while stack:
        yy, xx = stack.pop()
        if not (0 <= yy < height and 0 <= xx < width) or out[yy, xx] or not mask[yy, xx]:
            continue
        out[yy, xx] = True
        stack.extend(((yy + 1, xx), (yy - 1, xx), (yy, xx + 1), (yy, xx - 1)))
    return out


def measure_payload_box(
    xyz: np.ndarray,
    seed_uv: Tuple[int, int],
    top_normal: Sequence[float],
    top_point: Sequence[float],
    margin_m: float = 0.005,
    min_height_m: float = 0.004,
    max_height_m: float = 0.30,
    max_span_m: float = 0.45,
    min_pixels: int = 60,
) -> Optional[PayloadBox]:
    """
    Parçanın yönlü sınırlayıcı kutusunu ölçer. Ölçemezse None.

    top_normal / top_point: parçanın ÜST yüzüne oturtulmuş düzlem
    (fit_surface'ten gelir), bulutun frame'inde.

    YÜKSEKLİK NASIL ALINIYOR: üst yüzün ve destek düzleminin BASKIN
    seviyeleri (histogram tepesi) arasındaki fark. Önce yüksek bir yüzdelik
    (p98) kullanılmıştı; ölçüldü ki bu gürültüyü sistematik olarak yukarı
    yanlıyor - 2 mm'lik piksel gürültüsünde 30 mm'lik kutu 35 mm ölçüldü,
    çünkü p98 yaklaşık 2σ'ya denk geliyor. Tepe (mode) yansız.

    Bunun bedeli: DÜZ OLMAYAN bir üst yüzde, baskın yüzeyin üstünde kalan
    çıkıntılar kutuya girmez. Bu hattın zaten kabul ettiği bir varsayım -
    is_graspable düzlemsellik artığı 4 mm'yi aşan yüzeyi reddediyor, yani
    kavradığımız her şeyin üstü tanım gereği düz. Yine de çıkıntı ihtimali
    varsa margin_m ile karşılanmalı.
    """
    normal = np.asarray(top_normal, dtype=np.float64)
    length = float(np.linalg.norm(normal))
    if length < 1e-9:
        return None
    normal = normal / length
    origin = np.asarray(top_point, dtype=np.float64)

    finite = np.isfinite(xyz).all(axis=2)
    if not finite.any():
        return None

    # Üst yüz düzlemine göre işaretli yükseklik (normal kameraya bakıyor).
    heights = np.full(xyz.shape[:2], np.nan)
    heights[finite] = (xyz[finite] - origin) @ normal

    support = find_support_plane(heights[finite], min_thickness=min_height_m)
    if support is None:
        return None
    thickness = float(-support)
    if not (min_height_m <= thickness <= max_height_m):
        return None

    # Destek düzleminin ÜSTÜNDEKİ pikseller (yarı yükseklik eşiği: gürültülü
    # zeminin karışmaması ve alçak parçanın kaybolmaması arasında denge).
    threshold = support + 0.5 * thickness
    mask = finite & (heights > threshold)
    component = _connected_mask(mask, seed_uv)
    if component is None:
        return None
    pixels = int(component.sum())
    if pixels < min_pixels:
        return None

    points = xyz[component]
    local_heights = heights[component]

    # Düzlem içi 2B dağılım -> PCA -> yönlü kutu
    first, second = _plane_basis(normal)
    planar = np.column_stack((points @ first, points @ second))
    centre2d = planar.mean(axis=0)
    centred = planar - centre2d
    try:
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    major, minor = vh[0], vh[1]
    span_major = centred @ major
    span_minor = centred @ minor
    extent_major = float(span_major.max() - span_major.min())
    extent_minor = float(span_minor.max() - span_minor.min())
    if extent_major <= 0.0 or extent_minor <= 0.0:
        return None

    # YATAY AKLI BAŞINDALIK SINIRI. Yükseklik iki yönden de sınırlanmıştı ama
    # ayak izi hiç sınırlanmamıştı, ve kaçak tam oradan oluyor.
    #
    # 17 Ağu 2026 koşusu: ölçüm 881x347x4 mm, 68219 piksel - görüntünün
    # %31'i. Bu bir parça değil, KONVEYÖR BANDININ KENDİSİ. Üst düzlem
    # parçaya değil banda oturunca destek düzlemi 4 mm aşağıda "bulunuyor",
    # eşik bandın 2 mm altına düşüyor ve bağlı bileşen bütün banda yayılıyor.
    #
    # Bedeli sessiz değil ama gecikmeli: 891x357 mm'lik kutu emme kabına
    # iliştiriliyor ve o andan sonra HİÇBİR plan üretilemiyor - kartezyen de
    # serbest de %0 dönüyor. Belirti "robot parçayı kaldıramadı" olarak
    # görünüyor, sebebi ise iki adım önceki ölçüm.
    #
    # Vakum kabının taşıyabileceğinden büyük bir ölçüm, ölçümün kendisinin
    # başarısız olduğunun kanıtıdır. None dönmek çağıranı payload_size
    # config'ine düşürür - hatalı devasa kutuyla devam etmekten her koşulda
    # iyidir.
    if max_span_m > 0.0 and max(extent_major, extent_minor) > max_span_m:
        return None

    top = _histogram_mode(local_heights, 0.002)
    if top is None:
        top = float(np.median(local_heights))
    measured_height = float(top - support)
    if not (min_height_m <= measured_height <= max_height_m):
        return None

    # Kutunun merkezi. first/second/normal ortonormal olduğu için üç bileşen
    # bağımsız yazılabiliyor:
    #   düzlem içi : PCA kutusunun orta noktası (mutlak izdüşüm)
    #   normal     : destek ile üst yüzün tam ortası
    mid_major = 0.5 * float(span_major.max() + span_major.min())
    mid_minor = 0.5 * float(span_minor.max() + span_minor.min())
    centre_planar = centre2d + mid_major * major + mid_minor * minor
    along_normal = float(origin @ normal) + support + 0.5 * measured_height
    centre = (
        centre_planar[0] * first
        + centre_planar[1] * second
        + along_normal * normal
    )

    axis_major = major[0] * first + major[1] * second
    axis_minor = minor[0] * first + minor[1] * second
    axes = np.column_stack((axis_major, axis_minor, normal))
    # Sağ el sistemi olsun (MoveIt kuaterniyonu det=+1 bekler)
    if float(np.linalg.det(axes)) < 0.0:
        axes[:, 1] *= -1.0

    measured = (extent_major, extent_minor, measured_height)
    inflated = tuple(float(v + 2.0 * margin_m) for v in measured)

    return PayloadBox(
        size=inflated,  # type: ignore[arg-type]
        size_measured=measured,  # type: ignore[arg-type]
        centre=(float(centre[0]), float(centre[1]), float(centre[2])),
        axes=axes,
        pixels=pixels,
        support_distance=thickness,
    )
