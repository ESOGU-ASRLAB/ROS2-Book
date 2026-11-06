#!/usr/bin/env python3
"""
Gemini algı düğümü: doğal dil sorgusu -> world frame'inde 3D tespitler + RViz marker.

Robotu hareket ettirmez. Hareketten önce "ER 2 sim görüntüsünde doğru yeri
gösteriyor mu" sorusunu ayrı ayrı doğrulamak için var.

Kullanım:
    ros2 topic pub --once /gemini/query std_msgs/String "{data: 'the red cube'}"
    ros2 topic echo /gemini/detections

RViz'de /gemini/markers (MarkerArray) eklenirse tespitler küre + etiket olarak görünür.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Any

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from .er_client import make_er_client
from .locator import GeminiLocator
from .markers import CYAN, DetectionMarkers


class GeminiPerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("gemini_perception")

        self.declare_parameter("backend", "gemini")
        self.declare_parameter("model", "gemini-robotics-er-2-preview")
        self.declare_parameter("thinking_level", "high")
        self.declare_parameter("jpeg_quality", 90)
        self.declare_parameter("vertex_project", "")
        self.declare_parameter("vertex_location", "us-central1")

        self.declare_parameter("image_topic", "/sim/image")
        self.declare_parameter("cloud_topic", "/sim/pointcloud")
        self.declare_parameter("depth_topic", "/sim/depth/image")
        self.declare_parameter("camera_info_topic", "/sim/camera_info")
        self.declare_parameter("intensity_topic", "/intensity")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("camera_frame_override", "")
        self.declare_parameter("deprojection", "auto")
        self.declare_parameter("sample_window", 5)
        self.declare_parameter("max_items", 10)
        self.declare_parameter("marker_lifetime_sec", 0.0)

        self.declare_parameter("er_image_source", "render")
        self.declare_parameter("render_mode", "normals")
        self.declare_parameter("render_min_m", 0.3)
        self.declare_parameter("render_max_m", 4.0)
        # Tamsayı derinliğin metre/LSB ölçeği. "16UC1 = milimetre" EVRENSEL
        # DEĞİL: SICK Visionary-T Mini 0.25 mm/LSB kullanıyor (mode_real.yaml).
        self.declare_parameter("depth_scale_m", 0.001)
        # Derinlik ışın boyu MESAFE mi (SICK ToF), yoksa optik eksene izdüşüm
        # Z mi (Gazebo, RealSense)? Bkz. depth_render.radial_to_planar.
        self.declare_parameter("depth_is_radial", False)
        self.declare_parameter("publish_er_image", True)
        self.declare_parameter("er_image_rate_hz", 2.0)
        self.declare_parameter("gripper_type", "vacuum")

        self.declare_parameter("patch_radius_px", 12)
        self.declare_parameter("patch_depth_band", 0.05)
        self.declare_parameter("patch_min_points", 25)

        def get(name: str) -> Any:
            return self.get_parameter(name).value

        er_source = str(get("er_image_source"))
        render_mode = str(get("render_mode"))
        gripper_type = str(get("gripper_type"))

        er_client = make_er_client(
            backend=str(get("backend")),
            model=str(get("model")),
            thinking_level=str(get("thinking_level")),
            jpeg_quality=int(get("jpeg_quality")),
            project=str(get("vertex_project")),
            location=str(get("vertex_location")),
            context_key="rgb" if er_source == "rgb" else render_mode,
            tool="vacuum" if gripper_type == "vacuum" else "parallel",
        )

        self.locator = GeminiLocator(
            node=self,
            er_client=er_client,
            image_topic=str(get("image_topic")),
            cloud_topic=str(get("cloud_topic")),
            depth_topic=str(get("depth_topic")),
            camera_info_topic=str(get("camera_info_topic")),
            intensity_topic=str(get("intensity_topic")),
            world_frame=str(get("world_frame")),
            camera_frame_override=str(get("camera_frame_override")),
            deprojection=str(get("deprojection")),
            sample_window=int(get("sample_window")),
            er_image_source=er_source,
            render_mode=render_mode,
            render_min_m=float(get("render_min_m")),
            render_max_m=float(get("render_max_m")),
            depth_scale_m=float(get("depth_scale_m")),
            depth_is_radial=bool(get("depth_is_radial")),
            publish_er_image=bool(get("publish_er_image")),
            er_image_rate_hz=float(get("er_image_rate_hz")),
            patch_radius_px=int(get("patch_radius_px")),
            patch_depth_band=float(get("patch_depth_band")),
            patch_min_points=int(get("patch_min_points")),
        )

        self.world_frame = str(get("world_frame"))
        self.max_items = int(get("max_items"))
        self.marker_lifetime_sec = float(get("marker_lifetime_sec"))

        self.detections_pub = self.create_publisher(String, "/gemini/detections", 10)
        # Namespace ayrı: pick_place_node aynı topic'e görev marker'ları yazar
        # ve iki düğüm birbirinin çizimini silmemeli (bkz. markers.py).
        self.markers = DetectionMarkers(
            node=self,
            world_frame=self.world_frame,
            namespace="gemini_query",
            lifetime_sec=self.marker_lifetime_sec,
        )
        self.create_subscription(String, "/gemini/query", self._on_query, 10)

        # Aynı anda tek sorgu: ER çağrısı saniyeler sürebiliyor ve topic'e arka
        # arkaya iki mesaj düşerse kota boşa gider.
        self._busy = threading.Lock()

        self.get_logger().info("gemini_perception hazır. Sorgu: /gemini/query (std_msgs/String)")

    def _on_query(self, msg: String) -> None:
        query = msg.data.strip()
        if not query:
            return
        if not self._busy.acquire(blocking=False):
            self.get_logger().warn(f"Önceki sorgu sürüyor, atlandı: {query!r}")
            return
        threading.Thread(target=self._run_query, args=(query,), daemon=True).start()

    def _run_query(self, query: str) -> None:
        try:
            detections = self.locator.locate(query, max_items=self.max_items)

            payload = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "frame_id": self.world_frame,
                "count": len(detections),
                "detections": detections,
            }
            out = String()
            out.data = json.dumps(payload, ensure_ascii=False)
            self.detections_pub.publish(out)
            self.markers.publish(detections, colour=CYAN)
        finally:
            self._busy.release()


def main() -> None:
    rclpy.init()
    node = GeminiPerceptionNode()
    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
