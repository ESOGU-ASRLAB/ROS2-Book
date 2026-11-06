#!/usr/bin/env python3
"""
Derinliğin BİRİMİ ve KONVANSİYONU - sim ile gerçek arasındaki asıl fark.

Buradaki asıl soru şu: "ER 2'ye gönderilen kare sahneyi gerçekten gösteriyor
mu?" İki sabit, sim'de doğru gerçekte yanlış olduğu için bu sessizce bozulmuştu
ve belirti algı katmanında hiç görünmüyordu - yalnızca "model nesneyi bulamadı"
diye çıkıyordu:

  1. ÖLÇEK. SICK sürücüsü ham distance map'i ölçeklemeden /depth'e koyuyor
     (visionary_t_mini.cpp:86-91) ve o haritanın birimi 0.25 mm
     (VisionaryTMiniData.cpp:41). 1000'e bölmek derinliği 4 kat büyütüyordu;
     render'ın 0.3-4.0 m geçerlilik penceresinden 217088 pikselin 15417'si
     (%7.1) geçiyor, ER'ye %93'ü siyah bir kare gidiyordu.

  2. KONVANSİYON. SICK bir ToF ve IŞIN BOYU mesafe yayınlıyor
     (VisionaryTMiniData.cpp:272 -> RADIAL), oysa pinhole açılımı derinliğin
     optik eksene izdüşüm (Z) olduğunu varsayar. Radyali Z sanmak düz bir
     duvarı kâseye çevirir.

Gazebo ikisini de doğru yaptığı (32FC1/metre/planar) için sim'de test etmek
imkânsızdı. Testler bu yüzden sentetik: bilinen bir düzlem üretilip iki yoldan
da geçiriliyor.

Çalıştırma:
    python3 -m pytest test/test_depth_conventions.py -v
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensor_msgs.msg import CameraInfo, Image  # noqa: E402

from ch09_conveyor_pick_place.depth_render import (  # noqa: E402
    SICK_T_MINI_DEPTH_SCALE_M, depth_to_metres, depth_to_points, radial_to_planar,
    render, _valid_mask,
)

# Gerçek SICK Visionary-T Mini V3S145-1A'nın canlı /camera_info'su
# (13 Ağu 2026, ur10e_sick_optical_frame).
WIDTH, HEIGHT = 512, 424
FX, FY = 365.666999, 365.819
CX, CY = 253.925999, 207.296999


def _camera_info() -> CameraInfo:
    info = CameraInfo()
    info.width, info.height = WIDTH, HEIGHT
    info.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
    return info


def _depth_msg_16u(raw: np.ndarray) -> Image:
    msg = Image()
    msg.height, msg.width = raw.shape
    msg.encoding = "16UC1"
    msg.step = msg.width * 2
    msg.data = raw.astype(np.uint16).tobytes()
    return msg


def _cos_theta() -> np.ndarray:
    """Her piksel için optik eksenle yapılan açının kosinüsü."""
    a = (np.arange(WIDTH, dtype=np.float32)[None, :] - CX) / FX
    b = (np.arange(HEIGHT, dtype=np.float32)[:, None] - CY) / FY
    return 1.0 / np.sqrt(a * a + b * b + 1.0)


def _synthetic_wall(distance_m: float) -> np.ndarray:
    """
    Kameraya tam dik, `distance_m` uzaklıkta bir duvarın RADYAL mesafe haritası.

    Planar Z her yerde sabit; radyal mesafe eksenden uzaklaştıkça Z/cos(theta)
    ile büyür. Gerçek sensörün yaydığı şey budur.
    """
    return distance_m / _cos_theta()


# --- 1. ÖLÇEK ---------------------------------------------------------------


def test_scale_is_not_hardcoded_to_millimetres():
    """0.25 mm/LSB verildiğinde metre değeri 1 mm varsayımının dörtte biri olmalı."""
    raw = np.full((HEIGHT, WIDTH), 7202, dtype=np.uint16)  # canlı karenin medyanı
    msg = _depth_msg_16u(raw)

    as_mm = depth_to_metres(msg, scale_m=0.001)
    as_sick = depth_to_metres(msg, scale_m=SICK_T_MINI_DEPTH_SCALE_M)

    assert SICK_T_MINI_DEPTH_SCALE_M == 0.00025
    assert np.allclose(as_mm, 7.202)
    assert np.allclose(as_sick, 1.8005)
    assert np.allclose(as_mm, as_sick * 4.0)


def test_wrong_scale_blanks_the_validity_window():
    """
    Asıl arıza: yanlış ölçek sahneyi geçerlilik penceresinin DIŞINA atıyor.

    Bu, "%93'ü siyah kare" belirtisinin ta kendisi. Ölçek hatası burada
    sessizce bir GÖRÜNÜRLÜK hatasına dönüşüyor - sayı yanlış diye değil,
    maske sahneyi tamamen elediği için.
    """
    raw = (_synthetic_wall(1.8) / SICK_T_MINI_DEPTH_SCALE_M).astype(np.uint16)
    msg = _depth_msg_16u(raw)

    wrong = _valid_mask(depth_to_metres(msg, scale_m=0.001), 0.3, 4.0)
    right = _valid_mask(depth_to_metres(msg, scale_m=SICK_T_MINI_DEPTH_SCALE_M), 0.3, 4.0)

    assert wrong.mean() < 0.10, "yanlış ölçekte sahnenin neredeyse tamamı elenmeli"
    assert right.mean() > 0.99, "doğru ölçekte sahnenin tamamı görünmeli"


# --- 2. RADYAL vs PLANAR ----------------------------------------------------


def test_radial_to_planar_flattens_a_wall():
    """Düz bir duvar, dönüşümden sonra her pikselde AYNI Z'yi vermeli."""
    radial = _synthetic_wall(1.8).astype(np.float32)

    assert radial.max() / radial.min() == pytest.approx(1.0 / _cos_theta().min(), rel=1e-3)

    planar = radial_to_planar(radial, _camera_info())
    assert np.allclose(planar, 1.8, atol=1e-4)


def test_treating_radial_as_planar_bulges_the_scene():
    """
    Radyali Z sanmak duvarı kâseye çeviriyor; kabarma ölçülebilir olmalı.

    Bu kamerada köşe açısı 41.9 derece, yani köşede 1/cos = 1.343: 1.8 m'lik
    sahnede 0.6 m yapay yükseklik. relief'in yükseklik penceresi 125 mm ve
    normals'ın yüzey normalleri bunun yanında tamamen anlamsız kalır.
    """
    info = _camera_info()
    radial = _synthetic_wall(1.8).astype(np.float32)

    corner_deg = math.degrees(math.acos(float(_cos_theta()[0, 0])))
    assert corner_deg == pytest.approx(41.9, abs=0.2)

    bulged = depth_to_points(radial, info)[:, :, 2]  # HATALI yol
    flat = depth_to_points(radial_to_planar(radial, info), info)[:, :, 2]

    assert bulged.max() - bulged.min() > 0.55, "kabarma yarım metreyi geçmeli"
    assert flat.max() - flat.min() < 1e-3, "düzeltilmiş duvar düz olmalı"


def test_render_applies_both_corrections():
    """render() uçtan uca: SICK ayarlarıyla düz duvar gerçekten düz çıkmalı."""
    raw = (_synthetic_wall(1.8) / SICK_T_MINI_DEPTH_SCALE_M).astype(np.uint16)
    msg, info = _depth_msg_16u(raw), _camera_info()

    frame = render(
        mode="relief", depth_msg=msg, camera_info=info,
        min_m=0.3, max_m=4.0, height_lo_m=-0.005, height_hi_m=0.12,
        depth_scale_m=SICK_T_MINI_DEPTH_SCALE_M, depth_is_radial=True,
    )
    assert frame is not None
    # relief düzleme göre yükseklik boyar; düz duvarın tamamı tek tonda olmalı.
    assert frame.reshape(-1, 3).std(axis=0).max() < 8.0

    broken = render(
        mode="relief", depth_msg=msg, camera_info=info,
        min_m=0.3, max_m=4.0, height_lo_m=-0.005, height_hi_m=0.12,
        depth_scale_m=0.001, depth_is_radial=False,
    )
    black = (broken.sum(axis=2) == 0).mean()
    assert black > 0.90, "düzeltmesiz hâlde karenin %90'ından fazlası siyah olmalı"


def test_radial_flag_requires_camera_info():
    """camera_info olmadan radyal düzeltme yapılamaz; sessizce yanlış render etme."""
    raw = np.full((HEIGHT, WIDTH), 7202, dtype=np.uint16)
    out = render(
        mode="gray", depth_msg=_depth_msg_16u(raw), camera_info=None,
        depth_scale_m=SICK_T_MINI_DEPTH_SCALE_M, depth_is_radial=True,
    )
    assert out is None


# --- 3. SİM PARİTESİ --------------------------------------------------------


def test_sim_path_is_untouched():
    """Gazebo 32FC1/metre/planar: ölçek uygulanmamalı, dönüşüm yapılmamalı."""
    msg = Image()
    msg.height, msg.width = 4, 4
    msg.encoding = "32FC1"
    msg.step = msg.width * 4
    msg.data = np.full((4, 4), 1.75, dtype=np.float32).tobytes()

    # 32FC1 zaten metrik; scale_m ne verilirse verilsin değişmemeli.
    assert np.allclose(depth_to_metres(msg, scale_m=0.001), 1.75)
    assert np.allclose(depth_to_metres(msg, scale_m=SICK_T_MINI_DEPTH_SCALE_M), 1.75)
