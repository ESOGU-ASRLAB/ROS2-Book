#!/usr/bin/env python3
"""
Derinlikten ER 2'ye verilecek görüntüyü üretir.

NEDEN: Gerçek SICK Visionary-T Mini V3S145-1A bir ToF kamerası ve RGB modu YOK.
Yayınladığı şeyler depth (16UC1, mm), intensity (16UC1, yakın-IR genlik),
statemap ve points. Gazebo'daki rgbd_camera ise RGB üretiyor ama gerçek
donanımda böyle bir görüntü hiç olmayacak. Sim'de RGB ile çalışıp gerçeğe
geçmek, üzerinde çalıştığımız modaliteyi tamamen değiştirmek demek.

Bu yüzden ER 2'ye verilen görüntü, iki dünyada da var olan tek şeyden
üretiliyor: DERİNLİK. Her iki taraf da önce metreye çevrilir, sonra AYNI sabit
metrik aralıkla (render_min_m .. render_max_m) normalize edilip aynı şekilde
render edilir. Parite buradan gelir - sim ve gerçek görüntüler birbirinin
yerine geçebilir olmalı, yoksa sim'de tutturduğunuz prompt gerçekte tutmaz.

Render modları:
  normals : derinlikten yüzey normali + Lambert gölgeleme. Mat gri bir 3B
            render gibi görünür; kenarlar, eğrilik ve düz yüzeyler doğru okunur.
            VLM'ler için en "doğal" olanı, varsayılan bu.
  turbo   : sabit aralıkta renklendirilmiş derinlik. Derinlik sıralamasını
            korur ama renkler yapay.
  gray    : sabit aralıkta ters normalize edilmiş gri derinlik. En yalın.
  intensity : kameranın kendi IR genlik görüntüsü (yalnızca gerçek donanımda
            var; sim'de yayınlanmaz). Gerçekte en "fotoğraf gibi" olan bu.

UYARI: hiçbiri ER 2 için dağılım-içi (in-distribution) girdi değil. Derinlikten
nesne KATEGORİSİ tanımak RGB'ye göre zayıf olacaktır. Buna karşılık geometrik
sorgular ("düz panel", "kutunun üstü", "en yakın nesne") iyi çalışır - ve vakumlu
kavrama zaten geometrik bir problem olduğu için görev bu modaliteye uygun.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sensor_msgs.msg import CameraInfo, Image

try:
    import cv2
    CV_AVAILABLE = True
except ImportError:  # pragma: no cover
    CV_AVAILABLE = False


RENDER_MODES = ("relief", "normals", "turbo", "gray", "intensity")


# 16 bit tamsayı derinlik için VARSAYILAN birim (metre / LSB). Çoğu RGB-D
# kamerası (RealSense, Kinect) milimetre yayınlar.
DEFAULT_DEPTH_SCALE_M = 0.001

# SICK Visionary-T Mini: ham distance map birimi 0.25 mm'dir, milimetre DEĞİL
# (sick_visionary_cpp_shared/src/VisionaryTMiniData.cpp:41,
#  DISTANCE_MAP_UNIT = 0.25f; sürücü ham uint16'yı ölçeklemeden /depth'e
#  koyuyor, ölçeği yalnızca kendi generatePointCloud'unda uyguluyor).
# ÖLÇÜLDÜ (13 Ağu 2026, canlı hücre): ham /depth ile aynı karenin /points
# bulutunun ||xyz||'si karşılaştırıldı, oran 4019.6 çıktı - yani 1/0.00025.
SICK_T_MINI_DEPTH_SCALE_M = 0.00025


def depth_to_metres(msg: Image, scale_m: float = DEFAULT_DEPTH_SCALE_M) -> Optional[np.ndarray]:
    """
    Depth Image mesajını metre cinsinden float32 diziye çevirir.

    Sim (Gazebo) 32FC1/metre yayınlıyor ve zaten metriktir; tamsayı encoding'ler
    ham LSB yayınlar ve `scale_m` (metre/LSB) ile çarpılır.

    scale_m ELLE VERİLMELİ: "16UC1 = milimetre" evrensel bir kural değil.
    SICK Visionary-T Mini 0.25 mm/LSB kullanıyor ve 1000'e bölmek derinliği
    4 KAT büyütür. Belirtisi sessizdir: render'ın geçerlilik penceresi
    (render_min_m..render_max_m) sahnenin neredeyse tamamını dışarıda bırakır ve
    ER 2'ye %93'ü siyah bir kare gider (ölçüldü: 0.3-4.0 m penceresinde
    217088 pikselin yalnızca 15417'si, %7.1'i hayatta kalıyordu; doğru ölçekle
    215224, yani %99.1).
    """
    if msg.encoding == "32FC1":
        depth = np.frombuffer(msg.data, dtype=np.float32).astype(np.float32)
    elif msg.encoding in ("16UC1", "mono16"):
        depth = np.frombuffer(msg.data, dtype=np.uint16).astype(np.float32) * float(scale_m)
    else:
        return None
    return depth.reshape(msg.height, msg.width)


def radial_to_planar(depth_m: np.ndarray, camera_info: CameraInfo) -> np.ndarray:
    """
    Işın boyu (radyal) mesafeyi pinhole Z derinliğine çevirir.

    NEDEN: bu dosyadaki pinhole açılımı (depth_to_points) derinliğin OPTİK EKSENE
    İZDÜŞÜM, yani Z olduğunu varsayar. SICK Visionary-T Mini ise ışın boyunca
    MESAFE yayınlar (VisionaryTMiniData.cpp:272 haritayı RADIAL olarak veriyor).
    Gazebo'nun depth kamerası planar Z yayınlar - yani bu, sim ile gerçek
    arasındaki asıl modalite farkı ve sim'de görünmez.

    Radyal veriyi Z sanmak sahneyi kameradan uzaklaşan bir KÂSEYE çevirir: hata
    ekseni terk ettikçe 1/cos(theta) ile büyür. ÖLÇÜLDÜ (canlı kare, fx=365.7,
    cx=253.9): köşe açısı 41.9 derece, yani köşede %34.3 fazla. 1.8 m'lik bir
    sahnede bu 0.6 m'lik yapay bir kabarma demek - relief'in 125 mm'lik yükseklik
    penceresini de, normals'ın yüzey normallerini de tamamen anlamsız kılar.

    ÖLÇÜLEN DOĞRULAMA: aynı karede |depth - ||xyz|| | medyanı 8.2 mm,
    |depth - z| medyanı 165 mm. Yani /depth gerçekten radyal.
    """
    height, width = depth_m.shape
    fx, fy = float(camera_info.k[0]), float(camera_info.k[4])
    cx, cy = float(camera_info.k[2]), float(camera_info.k[5])
    if fx == 0.0 or fy == 0.0:
        return depth_m

    if camera_info.width and camera_info.width != width:
        scale = width / float(camera_info.width)
        fx, cx = fx * scale, cx * scale
    if camera_info.height and camera_info.height != height:
        scale = height / float(camera_info.height)
        fy, cy = fy * scale, cy * scale

    us = np.arange(width, dtype=np.float32)[None, :]
    vs = np.arange(height, dtype=np.float32)[:, None]
    a = (us - cx) / fx
    b = (vs - cy) / fy
    return (depth_m / np.sqrt(a * a + b * b + 1.0)).astype(np.float32)


def _valid_mask(depth_m: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    return np.isfinite(depth_m) & (depth_m > min_m) & (depth_m < max_m)


def _normalize(depth_m: np.ndarray, valid: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    """
    Sabit metrik aralığı 0..1'e indirger (yakın = 1).

    Aralık SABİT olmalı: her kareyi kendi min/max'ına göre normalize edersek
    aynı sahne kamera hareket ettikçe farklı görünür ve sim/gerçek pariteyi
    kaybederiz.
    """
    clipped = np.clip(depth_m, min_m, max_m)
    norm = 1.0 - (clipped - min_m) / max(1e-6, (max_m - min_m))
    norm[~valid] = 0.0
    return norm


def render_gray(depth_m: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    valid = _valid_mask(depth_m, min_m, max_m)
    norm = _normalize(depth_m, valid, min_m, max_m)
    gray = (norm * 255.0).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def render_turbo(depth_m: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    valid = _valid_mask(depth_m, min_m, max_m)
    norm = _normalize(depth_m, valid, min_m, max_m)
    gray = (norm * 255.0).astype(np.uint8)
    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(gray, colormap)
    colored[~valid] = (0, 0, 0)  # geçersiz pikseller siyah
    return colored


def depth_to_points(depth_m: np.ndarray, camera_info: CameraInfo) -> np.ndarray:
    """Derinlik haritasını optik frame'de HxWx3 nokta dizisine açar (vektörel)."""
    height, width = depth_m.shape
    fx, fy = float(camera_info.k[0]), float(camera_info.k[4])
    cx, cy = float(camera_info.k[2]), float(camera_info.k[5])
    if fx == 0.0 or fy == 0.0:
        raise ValueError("CameraInfo.k boş (fx/fy sıfır)")

    # CameraInfo farklı çözünürlükteyse ana nokta ve odak uzaklığını ölçekle.
    if camera_info.width and camera_info.width != width:
        scale = width / float(camera_info.width)
        fx, cx = fx * scale, cx * scale
    if camera_info.height and camera_info.height != height:
        scale = height / float(camera_info.height)
        fy, cy = fy * scale, cy * scale

    us = np.arange(width, dtype=np.float32)[None, :]
    vs = np.arange(height, dtype=np.float32)[:, None]
    x = (us - cx) * depth_m / fx
    y = (vs - cy) * depth_m / fy
    return np.dstack((x, y, depth_m)).astype(np.float32)


def render_normals(
    depth_m: np.ndarray,
    camera_info: CameraInfo,
    min_m: float,
    max_m: float,
    light_dir: Tuple[float, float, float] = (-0.3, -0.5, 0.8),
    ambient: float = 0.25,
    stencil_px: int = 3,
    smooth: bool = True,
    smooth_sigma_m: float = 0.02,
) -> np.ndarray:
    """
    Derinlikten yüzey normali hesaplayıp Lambert gölgeleme uygular.

    Sonuç, sahnenin mat gri bir 3B render'ı gibi görünür. Renk bilgisi
    taşımaz - zaten sensörde de yok - ama biçim, kenar ve yönelim taşır.

    light_dir'in Z'si POZİTİF olmalı: bu vektör ışığın GİTTİĞİ yön ve optik
    frame'de +Z kameradan sahneye doğrudur. Negatif verilirse ışık sahnenin
    arkasından kameraya doğru gider ve kameraya bakan yüzeyler - yani ilgilenilen
    bütün yüzeyler - lambert=0 alıp ambient'te kalır (ölçüldü: ortalama parlaklık
    38/255, ER 2 hiçbir şey bulamıyor).

    smooth: normaller sonlu farkla hesaplandığı için derinlik gürültüsü doğrudan
    normale biner. SICK gürültü modelinin 5 mm'si, 2 m mesafede tek piksel farkında
    ~25 derece eğim hatası demek; yüzeyler zımpara gibi çıkar. Kenarları korumak
    için bilateral filtre + geniş stencil kullanılıyor.
    """
    valid = _valid_mask(depth_m, min_m, max_m)

    normals = surface_normals(
        depth_m, camera_info, valid,
        stencil_px=stencil_px, smooth=smooth, smooth_sigma_m=smooth_sigma_m,
    )

    light = np.asarray(light_dir, dtype=np.float32)
    light /= np.linalg.norm(light)
    lambert = np.clip(-(normals @ light), 0.0, 1.0)

    # Derinliğe göre hafif bir uzaklık soluklaştırması: aksi halde arka plandaki
    # düz duvar ile öndeki düz kutunun üstü aynı tonda çıkıyor ve ER 2 için
    # sahne düzleşiyor.
    depth_shade = _normalize(depth_m, valid, min_m, max_m)
    shade = ambient + (1.0 - ambient) * lambert
    shade = shade * (0.45 + 0.55 * depth_shade)

    gray = (np.clip(shade, 0.0, 1.0) * 255.0).astype(np.uint8)
    gray[~valid] = 0
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def surface_normals(
    depth_m: np.ndarray,
    camera_info: CameraInfo,
    valid: np.ndarray,
    stencil_px: int = 3,
    smooth: bool = True,
    smooth_sigma_m: float = 0.02,
) -> np.ndarray:
    """
    Piksel başına yüzey normali (HxWx3), hepsi kameraya doğru yönlendirilmiş.

    Normaller sonlu farkla hesaplandığı için derinlik gürültüsü doğrudan normale
    biner: SICK'in 5 mm'si tek piksel farkında ~25 derece eğim hatası demek.
    Kenarları korumak için bilateral filtre + geniş stencil kullanılıyor.
    """
    work = depth_m.astype(np.float32, copy=True)
    work[~valid] = 0.0
    if smooth:
        # Bilateral: düzlemleri düzleştirir ama derinlik sıçramalarını (nesne
        # kenarlarını) korur. sigmaColor metre cinsinden, çünkü görüntü metrik.
        work = cv2.bilateralFilter(work, d=7, sigmaColor=smooth_sigma_m, sigmaSpace=5)

    points = depth_to_points(np.where(valid, work, np.nan), camera_info)

    # Komşu farklarından teğet vektörler; normal = çapraz çarpım.
    step = max(1, int(stencil_px))
    dzdx = np.zeros_like(points)
    dzdy = np.zeros_like(points)
    dzdx[:, step:-step, :] = points[:, 2 * step:, :] - points[:, : -2 * step, :]
    dzdy[step:-step, :, :] = points[2 * step:, :, :] - points[: -2 * step, :, :]

    normals = np.cross(dzdx, dzdy)
    lengths = np.linalg.norm(normals, axis=2, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = normals / lengths
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)

    # Optik frame'de +Z kameradan uzağa bakar; hepsini kameraya çevir.
    normals[normals[:, :, 2] > 0.0] *= -1.0
    return normals


def _mode_of(values: np.ndarray, bins: int = 400) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    hist, edges = np.histogram(finite, bins=bins)
    return 0.5 * float(edges[hist.argmax()] + edges[hist.argmax() + 1])


def fit_dominant_plane(
    depth_m: np.ndarray,
    points: np.ndarray,
    valid: np.ndarray,
    camera_info: CameraInfo,
    band_m: float = 0.010,
    iterations: int = 2,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Baskın düzlemi (n, d) olarak döndürür: n.p + d, düzlemden YÜKSEKLİK.

    Normal KAMERAYA bakar, dolayısıyla düzlemin üstünde duran cisimler pozitif
    yükseklik alır.

    Yöntem: normal medyanından tohumla -> yükseklik histogramının modundan
    baskın yüzeyi seç -> sadece o yüzeye SVD ile uyum yap -> tekrarla.

    İki basit yaklaşım denendi ve ölçülerek elendi:

    1. Tüm geçerli piksellere doğrudan SVD: raylar ve arka plan uyumu çekiyor,
       normal 26 derece eğiliyor, MAD kırpması her turda daha kötüye yakınsıyor.
    2. Derinlik histogramının modundan tohumlama: kameraya dik bakışta mükemmel
       (gerçek karede 0.6 derece hata) ama kamera eğilince aynı derinlik
       dilimine farklı x'lerdeki farklı yüzeyler düşüyor ve tohum kirleniyor
       (25 derece eğimde düzlem bandın yerine RAYLARA kilitlendi).

    Normal medyanı eğimden bağımsız olduğu için tohum olarak sağlam; ama tek
    başına yetmez, çünkü bant/ray/kutu PARALEL yüzeyler - normal seçimi üçünü
    birden alır ve SVD ortalarından geçer. Yükseklik modu bu üçünü ayırır.
    """
    finite = valid & np.isfinite(depth_m)
    if finite.sum() < 200:
        return None

    normals = surface_normals(depth_m, camera_info, finite)
    good = finite & (np.linalg.norm(normals, axis=2) > 0.5)
    if good.sum() < 200:
        return None

    normal = np.median(normals[good], axis=0)
    norm_len = float(np.linalg.norm(normal))
    if norm_len < 1e-6:
        return None
    normal = normal / norm_len
    if normal[2] > 0.0:
        normal = -normal

    for _ in range(max(1, iterations)):
        with np.errstate(invalid="ignore"):
            height = points @ normal
        offset = -_mode_of(height[finite])
        sel = finite & np.isfinite(height) & (np.abs(height + offset) < band_m)
        if sel.sum() < 200:
            break
        chosen = points[sel]
        centroid = chosen.mean(axis=0)
        _, _, vt = np.linalg.svd(chosen - centroid, full_matrices=False)
        normal = vt[2]
        if normal[2] > 0.0:
            normal = -normal

    with np.errstate(invalid="ignore"):
        height = points @ normal
    return normal, -_mode_of(height[finite])


def render_relief(
    depth_m: np.ndarray,
    camera_info: CameraInfo,
    min_m: float,
    max_m: float,
    height_lo_m: float = -0.005,
    height_hi_m: float = 0.12,
) -> Optional[np.ndarray]:
    """
    Baskın düzleme göre YÜKSEKLİĞİ renklendirir.

    Neden gerek var: bant üstündeki 29 mm'lik bir kutunun üst yüzü bandınkiyle
    aynı normale sahiptir, yani `normals` modunda görünmez (ölçüldü: 1.7/255
    kontrast). Sabit metrik pencereli `turbo`/`gray` de 0.3-4.0 m aralığında
    29 mm'yi 2/255'e sıkıştırır. Düzlem çıkarılınca aynı kutu tüm renk
    skalasına yayılır.

    Ölçekten bağımsız: pencere kameranın yüzeye uzaklığına göre değil, yüzeyin
    KENDİSİNE göre tanımlı. Kamera yaklaşıp uzaklaşınca render değişmez ve
    gerçek SICK'te de aynı şekilde hesaplanır - sim/gerçek parite korunur.
    """
    valid = _valid_mask(depth_m, min_m, max_m)
    if not valid.any():
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)

    points = depth_to_points(depth_m, camera_info)
    plane = fit_dominant_plane(depth_m, points, valid, camera_info)
    if plane is None:
        return render_turbo(depth_m, min_m, max_m)
    normal, offset = plane

    # fit_dominant_plane normali kameraya döndürdüğü için bu doğrudan
    # "düzlemin üstündeki yükseklik". Geçersiz pikseller NaN üretir; clip
    # sonrası 0'a çekilip zaten siyah boyanıyor.
    with np.errstate(invalid="ignore"):
        height = points @ normal + offset
    height = np.nan_to_num(height, nan=height_lo_m, posinf=height_lo_m, neginf=height_lo_m)

    norm = np.clip((height - height_lo_m) / max(1e-6, height_hi_m - height_lo_m), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(gray, colormap)
    colored[~valid] = (0, 0, 0)
    return colored


def render_intensity(msg: Image, percentile_clip: float = 99.0) -> Optional[np.ndarray]:
    """
    SICK intensity (16UC1 IR genlik) görüntüsünü 8-bit BGR'ye çevirir.

    Genlik değerleri çok geniş dinamik aralığa yayılıyor ve birkaç parlak
    yansıma tüm görüntüyü karartıyor; bu yüzden üst yüzdelik ile kırpılır.
    Bu render YALNIZCA gerçek donanımda kullanılabilir - Gazebo rgbd_camera
    IR genlik üretmez.
    """
    if msg.encoding in ("16UC1", "mono16"):
        raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    elif msg.encoding == "mono8":
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
    else:
        return None

    finite = raw[raw > 0]
    if finite.size == 0:
        return np.zeros((msg.height, msg.width, 3), np.uint8)

    high = float(np.percentile(finite, percentile_clip))
    scaled = np.clip(raw.astype(np.float32) / max(1.0, high), 0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def render(
    mode: str,
    depth_msg: Optional[Image],
    camera_info: Optional[CameraInfo],
    intensity_msg: Optional[Image] = None,
    min_m: float = 0.3,
    max_m: float = 4.0,
    stencil_px: int = 3,
    smooth: bool = True,
    height_lo_m: float = -0.005,
    height_hi_m: float = 0.12,
    depth_scale_m: float = DEFAULT_DEPTH_SCALE_M,
    depth_is_radial: bool = False,
) -> Optional[np.ndarray]:
    """
    Seçilen modda ER 2'ye verilecek BGR görüntüyü üretir.

    height_lo_m / height_hi_m yalnız `relief` modunda kullanılır ve bakılan
    sahneye göre seçilmelidir: bant üstündeki bir kutu için düzlemin altını
    görmeye gerek yok, ama toolkit'te BOŞ GÖZ bir çukurdur - lo negatif
    olmazsa yüzeyle aynı tonda kalır (ölçüldü: 45 mm derin göz için
    lo=-0.005'te 28/255, lo=-0.08'de 50/255 kontrast).
    """
    if not CV_AVAILABLE:
        raise RuntimeError("OpenCV kurulu değil (python3-opencv).")

    if mode == "intensity":
        return render_intensity(intensity_msg) if intensity_msg is not None else None

    if depth_msg is None:
        return None
    depth_m = depth_to_metres(depth_msg, scale_m=depth_scale_m)
    if depth_m is None:
        return None

    # Radyal -> planar, HER MODDAN ÖNCE. gray/turbo için de gerekli: geçerlilik
    # penceresi metrik ve radyal mesafe eksen dışında sistematik olarak büyük.
    if depth_is_radial:
        if camera_info is None:
            return None
        depth_m = radial_to_planar(depth_m, camera_info)

    if mode == "relief":
        if camera_info is None:
            return None
        return render_relief(
            depth_m, camera_info, min_m, max_m,
            height_lo_m=height_lo_m, height_hi_m=height_hi_m,
        )
    if mode == "gray":
        return render_gray(depth_m, min_m, max_m)
    if mode == "turbo":
        return render_turbo(depth_m, min_m, max_m)
    if mode == "normals":
        if camera_info is None:
            return None
        return render_normals(
            depth_m, camera_info, min_m, max_m,
            stencil_px=stencil_px, smooth=smooth,
        )

    raise ValueError(f"Bilinmeyen render modu: {mode!r} ({', '.join(RENDER_MODES)})")
