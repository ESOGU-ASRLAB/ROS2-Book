#!/usr/bin/env python3
"""
Derinlikten parça boyutu ölçümünün bilinen bir sahnedeki doğruluğunu sınar.

Sentetik sahne kasten GERÇEĞE benzetildi: kamera eğik bakıyor, derinliğe
gürültü biniyor, parçanın yanında ikinci bir cisim duruyor ve zeminde
geçersiz (NaN) pikseller var. Bunların her biri ölçümü ayrı bir şekilde
bozabilir; test her birini ayrı ayrı yakalıyor.

    python3 -m pytest test/test_payload.py -v
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch09_conveyor_pick_place.payload import (  # noqa: E402
    find_support_plane, measure_payload_box,
)

# Gerçek RedCube ölçüsü
BOX = (0.140, 0.083, 0.030)


def make_scene(
    box_size=BOX, yaw_deg=0.0, noise_m=0.0, distance=0.6, resolution=(240, 320),
    fov_deg=60.0, second_box_at=None, invalid_fraction=0.0, seed=0,
):
    """
    Düz bir zemin ve üstünde bir kutu içeren organize bulut üretir.

    Kamera zemine TEPEDEN bakıyor: optik +Z ileri, zemin z=distance'ta.
    Dönen kutu, düzlem içi yönelimin ölçülüp ölçülmediğini sınamak için.
    """
    rng = np.random.default_rng(seed)
    height, width = resolution
    fx = fy = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cx, cy = width / 2.0, height / 2.0

    u = np.arange(width)[None, :].repeat(height, axis=0)
    v = np.arange(height)[:, None].repeat(width, axis=1)

    # Zemin düzlemi z = distance; kutu üstünde -> z = distance - h
    z = np.full((height, width), float(distance))

    def stamp(centre_xy, size, box_yaw):
        """
        Verilen XY merkezli kutunun üst yüzünü z haritasına işler.

        Metrik dönüşüm kutunun ÜST yüzünün derinliğinde yapılmalı, zeminde
        değil: perspektifte yakın yüzey büyük görünür, zemin ölçeğiyle seçilen
        piksel kümesi üst yüzde (distance-h)/distance kadar KÜÇÜK bir kutuya
        karşılık gelir. (İlk yazımda böyleydi ve 140 mm'lik kenar 131.6 mm
        ölçüldü - hata ölçümde değil, sahnedeydi.)
        """
        z_top = distance - size[2]
        x = (u - cx) / fx * z_top
        y = (v - cy) / fy * z_top
        dx, dy = x - centre_xy[0], y - centre_xy[1]
        c, s = math.cos(-box_yaw), math.sin(-box_yaw)
        rx, ry = c * dx - s * dy, s * dx + c * dy
        inside = (np.abs(rx) <= size[0] / 2.0) & (np.abs(ry) <= size[1] / 2.0)
        z[inside] = distance - size[2]
        return inside

    inside = stamp((0.0, 0.0), box_size, math.radians(yaw_deg))
    if second_box_at is not None:
        stamp(second_box_at, box_size, 0.0)

    if noise_m > 0:
        z += rng.normal(0.0, noise_m, z.shape)

    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    xyz = np.dstack((x, y, z)).astype(np.float64)

    if invalid_fraction > 0:
        bad = rng.random((height, width)) < invalid_fraction
        xyz[bad] = np.nan

    # ER'nin gösterdiği piksel: kutunun merkezi
    seed_uv = (int(cx), int(cy))
    # Üst yüz düzlemi: normal kameraya bakar (-Z), üzerinde bir nokta
    top_normal = (0.0, 0.0, -1.0)
    top_point = (0.0, 0.0, distance - box_size[2])
    return xyz, seed_uv, top_normal, top_point, inside


def measure(**kwargs):
    xyz, seed_uv, normal, point, _ = make_scene(**kwargs)
    return measure_payload_box(xyz, seed_uv, normal, point, margin_m=0.0)


def test_clean_scene_recovers_all_three_dimensions():
    box = measure()
    assert box is not None
    length, width, height = box.size_measured
    assert length == pytest.approx(BOX[0], abs=0.004), f"{length:.4f}"
    assert width == pytest.approx(BOX[1], abs=0.004), f"{width:.4f}"
    # Yükseklik en hassas ölçülen: iki uydurulmuş düzlem arasındaki fark.
    assert height == pytest.approx(BOX[2], abs=0.002), f"{height:.4f}"


@pytest.mark.parametrize("noise_m", [0.001, 0.002, 0.003, 0.005])
def test_height_survives_realistic_depth_noise(noise_m):
    """
    Piksel gürültüsü yüksekliği YANLAMAMALI.

    ASIL MESELE bu: yükseklik tek bir noktadan değil, YÜZLERCE pikselden çıkan
    iki seviyenin farkından geliyor. Yüzdelik alınırsa gürültü sistematik
    olarak yukarı yanlar (2 mm gürültüde 30 mm kutu 35 mm ölçülmüştü); tepe
    (mode) yansızdır ve gürültü büyüdükçe hata büyümez.
    """
    box = measure(noise_m=noise_m)
    assert box is not None
    assert box.size_measured[2] == pytest.approx(BOX[2], abs=0.003), \
        f"gurultu {noise_m * 1000:.0f} mm -> {box.size_measured[2] * 1000:.1f} mm"


def test_in_plane_orientation_is_recovered():
    """Düzlem içi yönelim ölçülüyor mu - elle sabitlemek zorunda kalmayalım."""
    for yaw in (0.0, 20.0, 35.0, 60.0):
        xyz, seed_uv, normal, point, _ = make_scene(yaw_deg=yaw)
        box = measure_payload_box(xyz, seed_uv, normal, point, margin_m=0.0)
        assert box is not None, yaw
        major = box.axes[:, 0]
        expected = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0])
        # Uzun kenar ekseni işaretten bağımsızdır (180 derece belirsizlik)
        error = math.degrees(math.acos(min(1.0, abs(float(major @ expected)))))
        assert error < 5.0, f"yaw={yaw}: {error:.1f} derece sapma"
        # Uzun kenar gerçekten UZUN olan olmalı
        assert box.size_measured[0] > box.size_measured[1]


def test_neighbouring_object_is_not_merged_into_the_box():
    """Yan yana duran ikinci kutu, bağlılık denetimi olmasa tek kutuya birleşirdi."""
    xyz, seed_uv, normal, point, _ = make_scene(second_box_at=(0.18, 0.0))
    box = measure_payload_box(xyz, seed_uv, normal, point, margin_m=0.0)
    assert box is not None
    # İki kutu birleşseydi uzunluk ~0.32 m çıkardı
    assert box.size_measured[0] < 0.18, box.size_measured


def test_missing_depth_pixels_are_tolerated():
    """Gerçek ToF kamerası delikli veri verir; ölçüm çökmemeli."""
    box = measure(invalid_fraction=0.15, noise_m=0.002)
    assert box is not None
    assert box.size_measured[2] == pytest.approx(BOX[2], abs=0.003)


def test_margin_inflates_every_side():
    xyz, seed_uv, normal, point, _ = make_scene()
    box = measure_payload_box(xyz, seed_uv, normal, point, margin_m=0.005)
    assert box is not None
    for measured, inflated in zip(box.size_measured, box.size):
        assert inflated == pytest.approx(measured + 0.010, abs=1e-9)


def test_no_object_returns_none_instead_of_a_bogus_box():
    """
    Düz zeminde ölçecek bir şey yoktur; ölçüm None dönmelidir.

    Uydurulmuş bir kutu, elle verilen yedekten DAHA tehlikelidir - kimse yanlış
    olduğunu fark etmez.
    """
    xyz, seed_uv, normal, point, _ = make_scene(box_size=(0.14, 0.083, 0.0))
    assert measure_payload_box(xyz, seed_uv, normal, point) is None


def test_object_taller_than_the_limit_is_rejected():
    xyz, seed_uv, normal, point, _ = make_scene(box_size=(0.14, 0.083, 0.05))
    assert measure_payload_box(
        xyz, seed_uv, normal, point, max_height_m=0.03) is None


def test_object_wider_than_the_limit_is_rejected():
    """
    Ayak izi sınırı. Yükseklik iki yönden sınırlıydı, yatay hiç değildi.

    Gerçek koşuda (17 Ağu 2026) ölçüm parçadan taşıp konveyör bandına yayıldı
    ve 881x347 mm çıktı. O kutu emme kabına iliştirilince hiçbir plan
    üretilemedi - kartezyen de serbest de %0 döndü - ve belirti iki adım
    sonra "robot parçayı kaldıramadı" olarak göründü. Uydurulmuş devasa bir
    kutu, elle verilen yedekten daha tehlikelidir.
    """
    xyz, seed_uv, normal, point, _ = make_scene(box_size=(0.30, 0.20, 0.03))
    assert measure_payload_box(
        xyz, seed_uv, normal, point, max_span_m=0.25) is None


def test_span_limit_does_not_reject_a_normal_part():
    xyz, seed_uv, normal, point, _ = make_scene(box_size=(0.14, 0.083, 0.03))
    assert measure_payload_box(
        xyz, seed_uv, normal, point, max_span_m=0.45) is not None


def test_span_limit_of_zero_disables_the_check():
    xyz, seed_uv, normal, point, _ = make_scene(box_size=(0.30, 0.20, 0.03))
    assert measure_payload_box(
        xyz, seed_uv, normal, point, max_span_m=0.0) is not None


def test_support_plane_is_the_mode_not_the_median():
    """Medyan, cismin pikselleriyle destek arasındaki boşluğa düşebilir."""
    # Eşit sayıda nokta: medyan tam iki tepenin ORTASINA, yani hiçbir yüzeyin
    # olmadığı boşluğa düşer. Gerçek karede de cisim kadraja hâkim olduğunda
    # olan budur.
    heights = np.concatenate([
        np.random.default_rng(0).normal(-0.030, 0.001, 700),  # destek
        np.random.default_rng(1).normal(0.000, 0.001, 700),  # üst yüz
    ])
    support = find_support_plane(heights)
    assert support == pytest.approx(-0.030, abs=0.002), support
    assert abs(np.median(heights) - (-0.030)) > 0.005, "medyan zaten doğruysa test anlamsız"
