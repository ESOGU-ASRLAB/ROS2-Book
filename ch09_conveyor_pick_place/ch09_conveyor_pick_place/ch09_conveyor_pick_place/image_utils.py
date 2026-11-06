#!/usr/bin/env python3
"""sensor_msgs/Image -> OpenCV BGR dönüşümü (cv_bridge bağımlılığı olmadan)."""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import Image

try:
    import cv2
    CV_AVAILABLE = True
except ImportError:  # pragma: no cover
    CV_AVAILABLE = False


def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    """
    Image mesajını BGR ndarray'e çevirir.

    Gazebo'nun kamera sensörü rgb8 yayınlıyor; kanal sırasını çevirmezsek
    kırmızı ile mavi takas olur. Yalnız er_image_source="rgb" yolunda kullanılır
    (gerçek SICK'te RGB yoktur).
    """
    if not CV_AVAILABLE:
        raise RuntimeError("OpenCV kurulu değil (python3-opencv).")

    if msg.encoding == "rgb8":
        array = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    if msg.encoding == "bgr8":
        return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()
    if msg.encoding == "rgba8":
        array = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    if msg.encoding == "bgra8":
        array = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    if msg.encoding == "mono8":
        array = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width)
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Desteklenmeyen encoding: {msg.encoding}")
