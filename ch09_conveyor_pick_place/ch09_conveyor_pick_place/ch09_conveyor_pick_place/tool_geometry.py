#!/usr/bin/env python3
"""
Uç eleman geometrisini URDF + mesh'ten ÖLÇER (elle girmek yerine).

NEDEN VAR: `tool_approach_vector` ve `tool_tip_offset` iki sabit ama ikisi de
tahmin edilecek şey değil, robotun kendi modelinde YAZILI. Elle girildiğinde
sessizce yanlış olabiliyorlar ve hatanın belirtisi ("kol parçaya değmiyor")
algı hatasına benziyor - oysa kamera doğru yeri gösteriyor, kol yanlış yere
gidiyor. 11 Ağu 2026'da tam bu oldu: tool_tip_offset 156 mm girilmişti, mesh'te
ölçülen gerçek değer 83.5 mm; kap her hedefin 72.6 mm üstünde duruyordu.

Bu modül iki soruyu mesh'e sorarak yanıtlar:

  1. Emme ekseni hangi yön?   -> dönel yüzey ekseni uydurulur
  2. Kabın AĞZI nerede?       -> eksen boyunca en uzak halkanın merkezi

Kap değişirse hiçbir sayı elle güncellenmez: mesh değişir, ölçüm değişir.

Kullanım:
    ros2 run ch09_conveyor_pick_place measure_tcp
    ros2 run ch09_conveyor_pick_place measure_tcp --link ur10e_suction_cup
    ros2 run ch09_conveyor_pick_place measure_tcp --urdf /tmp/robot.urdf

URDF varsayılan olarak /robot_state_publisher düğümünün robot_description
parametresinden okunur, yani MoveIt'in planladığı modelin TA KENDİSİNDEN.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]


# --- STL okuma -----------------------------------------------------------


def load_stl(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Bir STL dosyasını (N,3,3) üçgen ve (N,3) yüz normali olarak okur (binary + ascii)."""
    with open(path, "rb") as handle:
        header = handle.read(84)
        if len(header) < 84:
            raise ValueError(f"STL çok kısa: {path}")
        count = struct.unpack("<I", header[80:84])[0]
        body = handle.read()

    # Binary STL'in boyu tam olarak 84 + 50*N olmalı; değilse ascii'dir.
    if len(body) == count * 50 and count > 0:
        raw = np.frombuffer(body, dtype=np.uint8).reshape(count, 50)
        floats = raw[:, :48].copy().view(np.float32).reshape(count, 12)
        normals = floats[:, 0:3].astype(np.float64)
        tris = floats[:, 3:12].astype(np.float64).reshape(count, 3, 3)
        return tris, normals

    return _load_stl_ascii(path)


def _load_stl_ascii(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", errors="ignore") as handle:
        text = handle.read()
    normals = np.asarray(
        [[float(v) for v in m] for m in re.findall(
            r"facet\s+normal\s+(\S+)\s+(\S+)\s+(\S+)", text)],
        dtype=np.float64,
    )
    verts = np.asarray(
        [[float(v) for v in m] for m in re.findall(
            r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)],
        dtype=np.float64,
    )
    if len(verts) < 3 or len(verts) % 3 != 0:
        raise ValueError(f"Ascii STL çözülemedi: {path}")
    return verts.reshape(-1, 3, 3), normals


# --- URDF'ten link görselini bulma ---------------------------------------


@dataclass
class LinkVisual:
    mesh_path: str
    scale: Vec3
    origin_xyz: Vec3
    origin_rpy: Vec3


def _floats(text: Optional[str], default: Sequence[float]) -> Vec3:
    if not text:
        return tuple(float(v) for v in default)  # type: ignore[return-value]
    parts = [float(v) for v in text.replace(",", " ").split()]
    if len(parts) == 1:
        parts = parts * 3
    return (parts[0], parts[1], parts[2])


def _resolve_mesh_uri(uri: str) -> str:
    """file:// ve package:// URI'lerini yerel yola çevirir."""
    if uri.startswith("file://"):
        # file:////abs/path gibi fazladan eğik çizgiler URDF'lerde yaygın.
        return "/" + uri[len("file://"):].lstrip("/")
    if uri.startswith("package://"):
        pkg, _, rel = uri[len("package://"):].partition("/")
        try:
            from ament_index_python.packages import get_package_share_directory
            return os.path.join(get_package_share_directory(pkg), rel)
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"'{pkg}' paketi bulunamadı: {exc}") from exc
    return uri


def link_visual_mesh(urdf_xml: str, link_name: str) -> LinkVisual:
    """Verilen link'in URDF'teki görsel mesh'ini, ölçeğini ve origin'ini çıkarır."""
    root = ET.fromstring(urdf_xml)
    for link in root.iter("link"):
        if link.get("name") != link_name:
            continue
        for tag in ("visual", "collision"):
            for node in link.findall(tag):
                mesh = node.find("./geometry/mesh")
                if mesh is None:
                    continue
                origin = node.find("origin")
                return LinkVisual(
                    mesh_path=_resolve_mesh_uri(mesh.get("filename", "")),
                    scale=_floats(mesh.get("scale"), (1.0, 1.0, 1.0)),
                    origin_xyz=_floats(
                        origin.get("xyz") if origin is not None else None, (0, 0, 0)),
                    origin_rpy=_floats(
                        origin.get("rpy") if origin is not None else None, (0, 0, 0)),
                )
        raise ValueError(f"'{link_name}' link'inde mesh geometrisi yok")
    raise ValueError(f"URDF'te '{link_name}' link'i yok")


def _rpy_matrix(rpy: Vec3) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float64)


def vertices_in_link_frame(
    tris: np.ndarray, normals: np.ndarray, visual: LinkVisual
) -> Tuple[np.ndarray, np.ndarray]:
    """Mesh köşelerini/normallerini link frame'ine taşır (ölçek + origin)."""
    scale = np.asarray(visual.scale, dtype=np.float64)
    rotation = _rpy_matrix(visual.origin_rpy)
    verts = tris.reshape(-1, 3) * scale
    verts = verts @ rotation.T + np.asarray(visual.origin_xyz, dtype=np.float64)
    face_normals = normals @ rotation.T if len(normals) else normals
    return verts, face_normals


# --- dönel yüzey ekseni --------------------------------------------------


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        raise ValueError("sıfır uzunluklu vektör")
    return vector / length


def fit_revolution_axis(
    points: np.ndarray, normals: np.ndarray, iterations: int = 40
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Dönel yüzey eksenini (yön, eksen üstünde bir nokta, artık) uydurur.

    Kısıt: dönel bir yüzeyde her yüzey normali kendi noktası ve eksenle AYNI
    DÜZLEMDEDİR, yani  u . (n x (p - c)) = 0.  Bu (u, c) çiftinde çift-doğrusal
    olduğu için dönüşümlü çözülür:

      * c sabitken  ->  w_i = n_i x (p_i - c) alınır, u = min özvektör(sum w w^T)
      * u sabitken  ->  artık c'de doğrusaldır, en küçük kareler ile çözülür
          u.(n_i x p_i) - c.(u x n_i) = 0

    Bu yol neden: DENENİP ELENEN yöntemler (11 Ağu 2026 notu) - yüz normali
    kovaryansının en küçük özvektörü (çan yüzeyinde neredeyse izotrop çıkıyor:
    0.299 / 0.304 / 0.398), açık kenar halkası (mesh su geçirmez, açık kenar
    yok) ve "en geniş düz yüz" araması (montaj yüzünü buluyor, ağzı değil).
    """
    # points köşe başına (3N), normals yüz başına (N) gelir.
    if len(normals) == 0 or len(points) != 3 * len(normals) or len(normals) < 16:
        raise ValueError(
            f"eksen uydurmak için yeterli yüz normali yok "
            f"({len(points)} köşe, {len(normals)} normal)"
        )

    # Üçgen merkezleri: normal başına TEK nokta gerekir, köşe başına değil.
    centers = points.reshape(-1, 3, 3).mean(axis=1)
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 1e-9
    centers, unit_normals = centers[keep], normals[keep] / lengths[keep, None]

    # Başlangıç: mesh'in en uzun ekseni (PCA) - yön yaklaşık doğru olsun yeter.
    c = centers.mean(axis=0)
    _, _, vh = np.linalg.svd(centers - c, full_matrices=False)
    u = _unit(vh[0, :])

    residual = float("inf")
    for _ in range(iterations):
        # 1) c sabit -> u
        w = np.cross(unit_normals, centers - c)
        _, _, vh = np.linalg.svd(w, full_matrices=False)
        u = _unit(vh[2, :])

        # 2) u sabit -> c   (u.(n x p) = c.(u x n))
        a = np.cross(u, unit_normals)  # (M,3) katsayılar
        b = np.einsum("i,mi->m", u, np.cross(unit_normals, centers))
        solution, *_ = np.linalg.lstsq(a, b, rcond=None)
        c = solution

        new_residual = float(np.sqrt(np.mean(
            np.einsum("i,mi->m", u, np.cross(unit_normals, centers - c)) ** 2)))
        if abs(residual - new_residual) < 1e-12:
            residual = new_residual
            break
        residual = new_residual

    # Yönü mesh'in uzandığı tarafa çevir: kap, link orijininden DIŞARI bakar.
    if float(np.dot(points.mean(axis=0) - c, u)) < 0.0:
        u = -u
    return u, c, residual


# --- kap ağzı (TCP) ------------------------------------------------------


@dataclass
class TipMeasurement:
    axis: Vec3  # emme yönü, link frame'inde (birim)
    offset: Vec3  # link orijininden ağız merkezine vektör (m)
    distance: float  # |offset| (m)
    ring_radius: float  # ağız yarıçapı (m)
    ring_flatness: float  # ağız halkasının eksen boyunca std'si (m)
    ring_points: int
    axis_residual: float


def measure_tip(
    vertices: np.ndarray,
    axis: np.ndarray,
    axis_residual: float = 0.0,
    ring_band_m: float = 0.0005,
) -> TipMeasurement:
    """
    Eksen boyunca EN UZAK halkayı bulur; TCP o halkanın merkezidir.

    Neden "en uzak halka" ve "en uzak tek nokta" değil: ağız bir çember, tek
    köşe değil. Halkanın merkezi hem gürültüye dayanıklı, hem de eksen link
    orijininden geçmiyorsa doğru olan tek şey (o durumda TCP, s*u DEĞİLDİR).

    ring_flatness bir DOĞRULAMA: ağız düzlemi eksene dik olmalı. Bu sayı
    büyükse ya eksen yanlış uydurulmuştur ya da en uzak yüzey ağız değildir.
    """
    axis = _unit(np.asarray(axis, dtype=np.float64))
    s = vertices @ axis
    ring = vertices[s >= s.max() - ring_band_m]
    tip = ring.mean(axis=0)

    radial = ring - np.outer(ring @ axis, axis)
    # Halkanın kendi merkezine göre yarıçapı (eksen orijinden geçmeyebilir).
    radial_center = radial.mean(axis=0)
    radius = float(np.linalg.norm(radial - radial_center, axis=1).mean())

    return TipMeasurement(
        axis=tuple(float(v) for v in axis),  # type: ignore[arg-type]
        offset=tuple(float(v) for v in tip),  # type: ignore[arg-type]
        distance=float(np.linalg.norm(tip)),
        ring_radius=radius,
        ring_flatness=float((ring @ axis).std()),
        ring_points=int(len(ring)),
        axis_residual=float(axis_residual),
    )


def measure_from_urdf(urdf_xml: str, link_name: str) -> TipMeasurement:
    visual = link_visual_mesh(urdf_xml, link_name)
    tris, normals = load_stl(visual.mesh_path)
    vertices, face_normals = vertices_in_link_frame(tris, normals, visual)
    axis, _, residual = fit_revolution_axis(vertices, face_normals)
    return measure_tip(vertices, axis, residual)


# --- CLI -----------------------------------------------------------------


def _robot_description_from_ros(node_name: str) -> str:
    import rclpy
    from rcl_interfaces.srv import GetParameters

    rclpy.init()
    try:
        node = rclpy.create_node("measure_tcp_client")
        client = node.create_client(GetParameters, f"{node_name}/get_parameters")
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(
                f"{node_name} parametre servisi yok - robot ayakta mı? "
                "(--urdf ile dosyadan da okuyabilirsiniz)"
            )
        future = client.call_async(
            GetParameters.Request(names=["robot_description"]))
        rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
        response = future.result()
        if response is None or not response.values:
            raise RuntimeError("robot_description okunamadı")
        return response.values[0].string_value
    finally:
        rclpy.shutdown()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Uç eleman emme eksenini ve TCP'sini URDF mesh'inden ölçer."
    )
    parser.add_argument("--link", default="ur10e_suction_cup")
    parser.add_argument("--urdf", default=None, help="URDF dosyası (yoksa ROS'tan)")
    parser.add_argument("--node", default="/robot_state_publisher")
    parser.add_argument(
        "--compare", nargs=3, type=float, metavar=("X", "Y", "Z"),
        default=None, help="config'teki tool_tip_offset ile karşılaştır",
    )
    args = parser.parse_args(argv)

    if args.urdf:
        with open(args.urdf, "r") as handle:
            urdf_xml = handle.read()
    else:
        urdf_xml = _robot_description_from_ros(args.node)

    visual = link_visual_mesh(urdf_xml, args.link)
    tris, normals = load_stl(visual.mesh_path)
    vertices, face_normals = vertices_in_link_frame(tris, normals, visual)
    axis, center, residual = fit_revolution_axis(vertices, face_normals)
    tip = measure_tip(vertices, axis, residual)

    print(f"link           : {args.link}")
    print(f"mesh           : {visual.mesh_path}")
    print(f"  ölçek        : {visual.scale}")
    print(f"  origin       : {visual.origin_xyz}  rpy {visual.origin_rpy}")
    print(f"  üçgen        : {len(tris)}")
    print()
    print("--- dönel yüzey ekseni ---")
    print(f"  yön          : ({axis[0]:+.6f}, {axis[1]:+.6f}, {axis[2]:+.6f})")
    print(f"  frame X'ten  : {math.degrees(math.acos(min(1.0, abs(axis[0])))):.2f} derece")
    print(f"  eksen noktası: ({center[0]:+.5f}, {center[1]:+.5f}, {center[2]:+.5f})")
    print(f"  artık        : {residual * 1000:.4f} mm")
    print()
    print("--- kap ağzı (TCP) ---")
    print(f"  halka        : {tip.ring_points} köşe, "
          f"yarıçap {tip.ring_radius * 1000:.2f} mm")
    print(f"  düzlemsellik : {tip.ring_flatness * 1000:.4f} mm "
          f"(eksene dik olmalı -> ~0)")
    print(f"  uzaklık      : {tip.distance * 1000:.2f} mm")
    print()
    if args.compare is not None:
        configured = np.asarray(args.compare, dtype=np.float64)
        delta = float(np.linalg.norm(configured) - tip.distance)
        print(f"config değeri  : {np.linalg.norm(configured) * 1000:.2f} mm")
        print(f"FARK           : {delta * 1000:+.2f} mm "
              f"({'kap hedefin ÜSTÜNDE kalır' if delta > 0 else 'kap hedefin İÇİNE girer'})")
        print()
    print("config/gemini_params.yaml içine:")
    print(f"    tool_approach_vector: [{axis[0]:.7f}, {axis[1]:.7f}, {axis[2]:.7f}]")
    print(f"    tool_tip_offset: [{tip.offset[0]:.6f}, {tip.offset[1]:.6f}, "
          f"{tip.offset[2]:.6f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
