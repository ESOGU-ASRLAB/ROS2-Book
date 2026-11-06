#!/usr/bin/env python3
"""
Uç eleman geometrisi ve kavrama zinciri testleri.

Buradaki asıl soru şu: "kabın AĞZI istenen noktaya geliyor mu?" Bu, üç ayrı
sabitin (emme ekseni, TCP uzaklığı, kavrama kuaterniyonu) BİRLİKTE doğru
olmasını gerektiriyor ve her biri tek başına doğru görünürken bileşke yanlış
olabiliyor - 11 Ağu 2026'da tam olarak bu oldu (TCP 156 mm girilmişti, gerçeği
84.2 mm; kap her hedefin 71.8 mm üstünde duruyordu). O yüzden test tek tek
sabitleri değil, uçtan uca MESAFEYİ ölçüyor.

Çalıştırma:
    python3 -m pytest test/test_tool_geometry.py -v
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch09_conveyor_pick_place.grasp import (  # noqa: E402
    approach_quaternion, fit_surface, quaternion_to_matrix,
)
from ch09_conveyor_pick_place.tool_geometry import (  # noqa: E402
    fit_revolution_axis, load_stl, measure_tip,
    vertices_in_link_frame,
)


# --- sentetik "kap": ekseni bilinen bir dönel yüzey ----------------------


def _make_cup(axis, length=0.0843, mouth_radius=0.020, shaft_radius=0.007):
    """
    Bilinen eksende bir sap + çan üretir (üçgenler + yüz normalleri).

    Gerçek mesh'e benzemesi gerekmiyor; TEK gereken dönel olması ve ağzının
    eksene dik bir halka olması - ölçüm yönteminin dayandığı iki özellik.
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    # eksene dik iki birim vektör
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(seed @ axis)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    rings = []
    for s, radius in ((0.0, shaft_radius), (length * 0.6, shaft_radius),
                      (length, mouth_radius)):
        angles = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
        rings.append(np.array([
            s * axis + radius * (math.cos(a) * e1 + math.sin(a) * e2)
            for a in angles
        ]))

    tris, normals = [], []
    for lower, upper in zip(rings[:-1], rings[1:]):
        n = len(lower)
        for i in range(n):
            j = (i + 1) % n
            for triangle in ((lower[i], lower[j], upper[i]),
                             (lower[j], upper[j], upper[i])):
                triangle = np.asarray(triangle)
                edge1 = triangle[1] - triangle[0]
                edge2 = triangle[2] - triangle[0]
                normal = np.cross(edge1, edge2)
                length_n = np.linalg.norm(normal)
                if length_n < 1e-12:
                    continue
                normal = normal / length_n
                # normal DIŞA baksın
                radial = triangle.mean(axis=0)
                radial = radial - float(radial @ axis) * axis
                if float(normal @ radial) < 0.0:
                    normal = -normal
                tris.append(triangle)
                normals.append(normal)
    return np.asarray(tris), np.asarray(normals)


def test_revolution_axis_recovered_from_synthetic_cup():
    """Eksen uydurma, bilinen ekseni geri bulmalı (30 derece eğik dahil)."""
    for truth in ((0.0, 0.0, 1.0),
                  (1.0, 0.0, 0.0),
                  (math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0)):
        tris, normals = _make_cup(truth)
        axis, _, _ = fit_revolution_axis(tris.reshape(-1, 3), normals)
        truth_unit = np.asarray(truth) / np.linalg.norm(truth)
        error_deg = math.degrees(math.acos(min(1.0, abs(float(axis @ truth_unit)))))
        assert error_deg < 0.5, f"{truth}: eksen {error_deg:.3f} derece kaydı"


def test_tip_is_the_mouth_ring_not_the_farthest_vertex():
    """TCP, ağız halkasının MERKEZİ olmalı; tek bir uç köşe değil."""
    axis = (math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0)
    tris, normals = _make_cup(axis, length=0.0843, mouth_radius=0.020)
    tip = measure_tip(tris.reshape(-1, 3), np.asarray(axis))

    assert tip.distance == pytest.approx(0.0843, abs=1e-4)
    assert tip.ring_radius == pytest.approx(0.020, abs=5e-4)
    # Ağız düzlemi eksene dik: halkanın eksen boyunca yayılımı ~0.
    assert tip.ring_flatness < 1e-6
    # Merkez eksende: yarıçap bileşeni sıfır olmalı.
    offset = np.asarray(tip.offset)
    radial = offset - float(offset @ np.asarray(axis)) * np.asarray(axis)
    assert float(np.linalg.norm(radial)) < 1e-6


# --- gerçek mesh (varsa) -------------------------------------------------


def _cup_stl_path() -> str:
    """
    Emme kabı ağını araştırma deposunun ur_description paketinde arar.

    ROS 2'nin resmî ur_description paketinde bu ağ yoktur; bulunamazsa boş
    metin döner ve test atlanır.
    """
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("ur_description")
    except Exception:
        return ""
    return os.path.join(share, "meshes", "ur10e", "collision", "suction_cup.stl")


CUP_STL = _cup_stl_path()
CUP_VISUAL_ORIGIN = (-1.19139, -0.578159, -0.71775)


@pytest.mark.skipif(not os.path.exists(CUP_STL), reason="mesh yok")
def test_real_cup_mesh_matches_configured_constants():
    """
    Yapılandırmadaki iki sabit, gerçek mesh'ten ölçülenle uyuşmalı.

    Bu test kırmızıya dönerse config yanlış demektir - tersi değil.
    """
    from ch09_conveyor_pick_place.tool_geometry import LinkVisual

    tris, normals = load_stl(CUP_STL)
    visual = LinkVisual(CUP_STL, (0.001, 0.001, 0.001), CUP_VISUAL_ORIGIN, (0, 0, 0))
    vertices, face_normals = vertices_in_link_frame(tris, normals, visual)
    axis, center, _ = fit_revolution_axis(vertices, face_normals)
    tip = measure_tip(vertices, axis)

    configured_axis = np.array([0.8660254, 0.5, 0.0])
    configured_tip = np.array([0.072920, 0.042100, 0.0])

    axis_deg = math.degrees(math.acos(min(1.0, abs(float(axis @ configured_axis)))))
    assert axis_deg < 0.5, f"emme ekseni config'ten {axis_deg:.2f} derece farklı"
    assert tip.distance == pytest.approx(
        float(np.linalg.norm(configured_tip)), abs=0.001
    ), f"TCP {tip.distance * 1000:.1f} mm ölçüldü"

    # Ağız gerçekten düz bir halka olmalı, yoksa "en uzak yüzey" ağız değildir.
    assert tip.ring_flatness < 0.0005
    assert 0.015 < tip.ring_radius < 0.025
    # Eksen link orijininden geçiyor: mesh, frame'in tam üstüne oturtulmuş.
    axis_offset = float(np.linalg.norm(center - float(center @ axis) * axis))
    assert axis_offset < 0.002


@pytest.mark.skipif(not os.path.exists(CUP_STL), reason="mesh yok")
def test_cup_mouth_reaches_the_target_surface():
    """
    UÇTAN UCA: hedefi frame'e çevirip kabın ağzını geri hesapla.

    Bu, pick_place_node._frame_position_for_tip'in tersini yapar. Doğru
    sabitlerle kalan hata mikron mertebesinde olmalı; 156 mm'lik eski değerle
    bu test 72 mm hata verirdi.
    """
    from ch09_conveyor_pick_place.tool_geometry import LinkVisual

    tris, normals = load_stl(CUP_STL)
    visual = LinkVisual(CUP_STL, (0.001, 0.001, 0.001), CUP_VISUAL_ORIGIN, (0, 0, 0))
    vertices, face_normals = vertices_in_link_frame(tris, normals, visual)
    axis, _, _ = fit_revolution_axis(vertices, face_normals)
    tip_offset = np.asarray(measure_tip(vertices, axis).offset)

    # Yatay bir bant yüzeyi (normal +Z) ve üstündeki hedef nokta
    for normal_world in ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)):
        target = np.array([0.750, 0.100, 0.865])
        quat = approach_quaternion(normal_world, approach_vector=axis)
        rotation = quaternion_to_matrix(quat)

        # düğümün yaptığı: frame = uç - R * offset
        frame_position = target - rotation @ tip_offset
        # fiziğin yaptığı: ağız = frame + R * offset
        mouth = frame_position + rotation @ tip_offset
        assert np.allclose(mouth, target, atol=1e-9)

        # ve kap yüzeye DİK gelmeli: emme yönü = -normal. Tolerans, mesh'ten
        # uydurulan eksenin kendi artığı kadar (0.01 derece mertebesinde);
        # konum eşitliği yukarıda cebirsel olarak birebir tutuyor.
        approach_world = rotation @ (tip_offset / np.linalg.norm(tip_offset))
        deviation_deg = math.degrees(math.acos(min(1.0, abs(
            float(approach_world @ -np.asarray(normal_world))))))
        assert deviation_deg < 0.05, f"{normal_world}: {deviation_deg:.4f} derece"


# --- derinlik bandı: konvansiyondan bağımsız mı? ------------------------


def _plane_patch(origin, normal, count=400, extent=0.02, seed=0):
    rng = np.random.default_rng(seed)
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    seed_vec = np.array([1.0, 0.0, 0.0])
    if abs(float(seed_vec @ normal)) > 0.9:
        seed_vec = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(normal, seed_vec)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    uv = rng.uniform(-extent, extent, size=(count, 2))
    return np.asarray(origin) + uv[:, :1] * e1 + uv[:, 1:] * e2


def test_depth_band_rejects_background_in_both_conventions():
    """
    Band, arka planı hem x-ileri hem z-ileri bulutta elemeli.

    Eskiden band z bileşenine bakıyordu. Optik frame'de (z=ileri) doğruydu,
    Gazebo'nun x-ileri bulutunda ise z YANAL bir eksen: filtre arka planı hiç
    elemiyor, düzlem iki yüzeyin arasına oturuyordu.
    """
    cases = {
        # (ön yüzey merkezi, arka plan merkezi, yüzey normali)
        "z-ileri (optik)": ((0.0, 0.0, 0.80), (0.0, 0.0, 1.60), (0.0, 0.0, -1.0)),
        "x-ileri (gazebo)": ((0.80, 0.0, 0.0), (1.60, 0.0, 0.0), (-1.0, 0.0, 0.0)),
    }
    for name, (front, back, normal) in cases.items():
        points = np.vstack([
            _plane_patch(front, normal, count=300, seed=1),
            _plane_patch(back, normal, count=300, seed=2),
        ])
        patch = fit_surface(points, depth_band=0.05, min_points=25, anchor=front)
        assert patch is not None, name
        # Arka plan tamamen elenmeli: 600 noktanın yarısı kalmalı.
        assert patch.inliers == 300, f"{name}: {patch.inliers} nokta kaldı"
        # Merkez ön yüzeyde olmalı, iki yüzeyin ortasında değil.
        assert np.allclose(patch.centroid, front, atol=0.002), name
