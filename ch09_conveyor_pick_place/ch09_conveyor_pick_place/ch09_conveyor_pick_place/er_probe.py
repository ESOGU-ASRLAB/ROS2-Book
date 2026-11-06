#!/usr/bin/env python3
"""
ER 2 bağlantı testi - ROS'suz.

API anahtarının çalıştığını, modelin erişilebilir olduğunu ve pointing
çıktısının beklenen formatta geldiğini Gazebo'yu açmadan doğrular.

    ros2 run ch09_conveyor_pick_place er_probe --query "the flat surfaces"

--image verilmezse derinlik + camera_info topic'lerinden tek kare üretir
(ROS gerekir). Sonuç, işaretlenmiş noktalarla birlikte er_probe_out.png
olarak kaydedilir. GEMINI_API_KEY tanımlı olmalıdır.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable, Optional, TYPE_CHECKING

import cv2

from .er_client import API_KEY_PATHS, load_api_key, make_er_client

if TYPE_CHECKING:  # yalnızca tip ipuçları için; çalışma anında içe aktarılmaz
    import numpy as np


def _grab_frame_from_ros(args: argparse.Namespace) -> Optional[np.ndarray]:
    """
    Topic'lerden tek kare yakalar ve ER'ye verilecek görüntüyü üretir.

    Varsayılan derinlik yoludur: gerçek SICK Visionary-T Mini'de RGB topic'i
    hiç yayınlanmaz, dolayısıyla bu sonda RGB'yi test etmek gerçek donanımda
    var olmayan bir yolu test etmek olurdu.
    """
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image

    from . import depth_render
    from .image_utils import imgmsg_to_bgr
    from .locator import SENSOR_QOS

    rclpy.init()
    node = Node("er_probe_grabber")
    holder = {}

    def want(key: str) -> Callable[[Any], Any]:
        return lambda m: holder.setdefault(key, m)

    if args.source == "rgb":
        node.create_subscription(Image, args.image_topic, want("image"), SENSOR_QOS)
        needed = {"image"}
    elif args.render_mode == "intensity":
        node.create_subscription(Image, args.intensity_topic, want("intensity"), SENSOR_QOS)
        needed = {"intensity"}
    else:
        node.create_subscription(Image, args.depth_topic, want("depth"), SENSOR_QOS)
        node.create_subscription(CameraInfo, args.info_topic, want("info"), SENSOR_QOS)
        needed = {"depth", "info"}

    deadline = node.get_clock().now().nanoseconds + int(args.timeout * 1e9)
    while rclpy.ok() and not needed <= holder.keys():
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.get_clock().now().nanoseconds > deadline:
            break

    missing = needed - holder.keys()
    frame = None
    if missing:
        print(f"HATA: şu veriler gelmedi: {', '.join(sorted(missing))}", file=sys.stderr)
    elif args.source == "rgb":
        frame = imgmsg_to_bgr(holder["image"])
    else:
        frame = depth_render.render(
            mode=args.render_mode,
            depth_msg=holder.get("depth"),
            camera_info=holder.get("info"),
            intensity_msg=holder.get("intensity"),
            min_m=args.render_min,
            max_m=args.render_max,
            depth_scale_m=args.depth_scale,
            depth_is_radial=args.depth_radial,
        )

    node.destroy_node()
    rclpy.shutdown()
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemini Robotics ER 2 bağlantı testi")
    parser.add_argument("--image", default="", help="Dosyadan görüntü (boşsa ROS topic'ten)")
    parser.add_argument("--source", default="render", choices=["render", "rgb"],
                        help="ER girdisi: derinlik render'ı (gerçek donanım) veya RGB (yalnız "
                             "sim)")
    parser.add_argument("--render-mode", default="normals",
                        choices=["relief", "normals", "turbo", "gray", "intensity"])
    parser.add_argument("--render-min", type=float, default=0.3)
    parser.add_argument("--render-max", type=float, default=4.0)
    # Gerçek SICK Visionary-T Mini için: --depth-scale 0.00025 --depth-radial
    # (bkz. config/mode_real.yaml). Varsayılanlar Gazebo'ya göredir.
    parser.add_argument("--depth-scale", type=float, default=0.001,
                        help="Tamsayı derinliğin metre/LSB ölçeği (SICK T Mini: 0.00025)")
    parser.add_argument("--depth-radial", action="store_true",
                        help="Derinlik ışın boyu mesafe (ToF), planar Z değil")
    parser.add_argument("--depth-topic", default="/sim/depth/image")
    parser.add_argument("--info-topic", default="/sim/camera_info")
    parser.add_argument("--intensity-topic", default="/intensity")
    parser.add_argument("--image-topic", default="/sim/image")
    parser.add_argument("--query", default="all objects on the table", help="Pointing sorgusu")
    parser.add_argument("--backend", default="gemini", choices=["gemini", "vertex"])
    parser.add_argument("--model", default="gemini-robotics-er-2-preview")
    parser.add_argument("--out", default="er_probe_out.png", help="İşaretli çıktı görüntüsü")
    parser.add_argument("--timeout", type=float, default=10.0, help="Topic bekleme süresi (s)")
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"HATA: görüntü okunamadı: {args.image}", file=sys.stderr)
            return 1
    else:
        print(f"ROS'tan kare bekleniyor ({args.timeout:.0f}s), kaynak={args.source}"
              + (f"/{args.render_mode}" if args.source == "render" else "") + "...")
        frame = _grab_frame_from_ros(args)
        if frame is None:
            return 1

    print(f"Görüntü: {frame.shape[1]}x{frame.shape[0]}  backend={args.backend}")
    # Anahtar ortamda YA DA ~/.config/gemini/api_key dosyasında olabilir;
    # yalnızca ortama bakmak yanlış uyarı veriyordu.
    if args.backend == "gemini" and not load_api_key():
        print(
            f"UYARI: Gemini API anahtarı bulunamadı ({API_KEY_PATHS[0]} ya da "
            "GEMINI_API_KEY).", file=sys.stderr,
        )

    client = make_er_client(
        backend=args.backend,
        model=args.model,
        context_key="rgb" if args.source == "rgb" else args.render_mode,
    )
    detections = client.point(frame, args.query)

    if not detections:
        print("Hiçbir tespit dönmedi.")
    for detection in detections:
        print(f"  {detection.label:30s} piksel=({detection.u}, {detection.v})")
        cv2.circle(frame, (detection.u, detection.v), 8, (0, 255, 255), 2)
        cv2.putText(
            frame, detection.label, (detection.u + 12, detection.v),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
        )

    cv2.imwrite(args.out, frame)
    print(f"İşaretli görüntü kaydedildi: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
