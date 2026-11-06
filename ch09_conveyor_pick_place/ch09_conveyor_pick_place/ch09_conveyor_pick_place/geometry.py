#!/usr/bin/env python3
"""
2D piksel -> 3D dünya koordinatı dönüşümleri.

ER 2 sadece piksel verir. Robotun ihtiyacı olan şey world frame'de bir noktadır.
Aradaki zincir:

    (u, v) --deprojeksiyon--> optik frame'de (x, y, z) --TF--> world

Deprojeksiyon için iki yol var, ikisi de destekleniyor:

  1. Organize PointCloud2 (tercih edilen). rgbd_camera sensörünün "points"
     bulutu 512x424 organize geldiği için piksel indeksi doğrudan noktaya
     karşılık geliyor; kamera matrisiyle uğraşmaya gerek kalmıyor.
  2. Depth image + CameraInfo. Bulut yayınlanmadığında ya da RGB ile depth
     çözünürlükleri farklı olduğunda devreye girer.

Her iki yolda da tekil piksel yerine küçük bir pencerenin medyanı alınıyor:
ER 2 nesnenin tam merkezine değil "üstüne" işaret eder, ve o tek piksel
nesnenin kenarına düşerse arka plandaki derinliği okuruz — sessizce metrelerce
sapmış bir kavrama pozu demek bu.
"""

from __future__ import annotations

import math
import struct
from typing import Any, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_geometry_msgs import do_transform_point


Point3 = Tuple[float, float, float]


def _field_offsets(cloud: PointCloud2) -> Optional[Tuple[int, int, int]]:
    offsets = {field.name: field.offset for field in cloud.fields}
    if not {"x", "y", "z"} <= offsets.keys():
        return None
    return (offsets["x"], offsets["y"], offsets["z"])


def point_from_cloud(
    cloud: PointCloud2, u: int, v: int, window: int = 5
) -> Optional[Point3]:
    """Organize buluttan (u, v) çevresindeki geçerli noktaların medyanını alır."""
    if cloud.height <= 1:
        return None  # organize değil; piksel indeksleme anlamsız
    offsets = _field_offsets(cloud)
    if offsets is None:
        return None
    x_off, y_off, z_off = offsets

    half = max(0, window // 2)
    samples = []
    for dv in range(-half, half + 1):
        for du in range(-half, half + 1):
            uu, vv = u + du, v + dv
            if not (0 <= uu < cloud.width and 0 <= vv < cloud.height):
                continue
            base = (vv * cloud.width + uu) * cloud.point_step
            try:
                x = struct.unpack_from("f", cloud.data, base + x_off)[0]
                y = struct.unpack_from("f", cloud.data, base + y_off)[0]
                z = struct.unpack_from("f", cloud.data, base + z_off)[0]
            except struct.error:
                continue
            if math.isnan(x) or math.isnan(y) or math.isnan(z) or math.isinf(z):
                continue
            samples.append((x, y, z))

    if not samples:
        return None
    array = np.asarray(samples, dtype=np.float64)
    return tuple(float(value) for value in np.median(array, axis=0))


def _depth_to_metres(image: Image, scale_m: float) -> Optional[np.ndarray]:
    """
    Depth Image mesajını metre cinsinden float32 diziye çevirir.

    Sim (32FC1/metre) ve tamsayı derinlik (scale_m metre/LSB) burada aynı birime
    iner; tek uygulama depth_render'da tutuluyor ki iki yol ayrışmasın.
    """
    from .depth_render import depth_to_metres

    return depth_to_metres(image, scale_m=scale_m)


def point_from_depth(
    depth_image: Image,
    camera_info: CameraInfo,
    u: int,
    v: int,
    window: int = 5,
    depth_scale_m: float = 0.001,
    depth_is_radial: bool = False,
) -> Optional[Point3]:
    """
    Pinhole modeliyle (u, v) + derinlikten optik frame'de 3D nokta üretir.

    (u, v) RGB görüntünün çözünürlüğünde gelir; depth farklı çözünürlükteyse
    oranlanarak ölçeklenir.

    depth_scale_m / depth_is_radial render yoluyla AYNI olmalı: gerçek SICK
    0.25 mm/LSB ve ışın boyu mesafe yayınlıyor (bkz. depth_render.py). Bu yol
    yalnız deprojection "depth"/"auto" iken kullanılır - varsayılan "cloud"da
    sürücünün kendi doğru bulutu kullanılır ve buraya hiç düşülmez.
    """
    depth = _depth_to_metres(depth_image, scale_m=depth_scale_m)
    if depth is None:
        return None
    if depth_is_radial:
        from .depth_render import radial_to_planar

        depth = radial_to_planar(depth, camera_info)

    fx, fy = camera_info.k[0], camera_info.k[4]
    cx, cy = camera_info.k[2], camera_info.k[5]
    if fx == 0.0 or fy == 0.0:
        return None

    # CameraInfo, RGB ile aynı çözünürlükte varsayılır; depth ondan farklıysa ölçekle.
    if camera_info.width and camera_info.width != depth_image.width:
        scale_u = depth_image.width / float(camera_info.width)
        scale_v = depth_image.height / float(camera_info.height)
        u_d, v_d = int(u * scale_u), int(v * scale_v)
    else:
        u_d, v_d = u, v

    half = max(0, window // 2)
    u_lo, u_hi = max(0, u_d - half), min(depth_image.width, u_d + half + 1)
    v_lo, v_hi = max(0, v_d - half), min(depth_image.height, v_d + half + 1)
    patch = depth[v_lo:v_hi, u_lo:u_hi]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None
    z = float(np.median(valid))

    # Deprojeksiyon CameraInfo'nun kendi piksel uzayında yapılmalı, depth'inkinde değil.
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return (x, y, z)


def transform_point(
    tf_buffer: Any,
    point: Point3,
    source_frame: str,
    target_frame: str,
    stamp: Any = None,
    timeout_sec: float = 0.5,
) -> Optional[Point3]:
    """
    Noktayı source_frame'den target_frame'e taşır; başarısızsa None.

    stamp verilmezse en son geçerli transform (Time()) kullanılır. Kol hareket
    halindeyken bilek kamerası için bu fark eder, o yüzden çağıran taraf
    görüntünün kendi damgasını geçirmelidir.
    """
    try:
        transform = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            stamp if stamp is not None else rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=timeout_sec),
        )
    except Exception:
        return None

    stamped = PointStamped()
    stamped.header.frame_id = source_frame
    stamped.point.x, stamped.point.y, stamped.point.z = (float(value) for value in point)
    result = do_transform_point(stamped, transform)
    return (result.point.x, result.point.y, result.point.z)


def rotate_vector(
    tf_buffer: Any,
    vector: Point3,
    source_frame: str,
    target_frame: str,
    stamp: Any = None,
    timeout_sec: float = 0.5,
) -> Optional[Point3]:
    """
    Yön vektörünü frame'ler arasında döndürür (öteleme UYGULANMAZ).

    Yüzey normalleri için şart: transform_point kullanılırsa normale kameranın
    konumu da eklenir ve elde edilen "normal" tamamen anlamsız olur.
    """
    try:
        transform = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            stamp if stamp is not None else rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=timeout_sec),
        )
    except Exception:
        return None

    q = transform.transform.rotation
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    vec = np.asarray(vector, dtype=np.float64)

    # v' = v + 2 * q_vec x (q_vec x v + w * v)
    q_vec = np.array([x, y, z], dtype=np.float64)
    t = 2.0 * np.cross(q_vec, vec)
    rotated = vec + w * t + np.cross(q_vec, t)
    return (float(rotated[0]), float(rotated[1]), float(rotated[2]))
