#!/usr/bin/env python3
"""
/gemini/markers üzerindeki RViz görselleştirmesi - İKİ düğüm de kullanır.

NEDEN PAYLAŞILIYOR: bu kod eskiden yalnızca perception_node içindeydi, yani
tespit küreleri ve normal okları SADECE /gemini/query ile geliyordu. Asıl
hareketi /gemini/command yaptığı için, görev sırasında RViz'de hiçbir şey
görünmüyordu - kolun nereye gittiğini gözle doğrulamanın tam da en gerekli
olduğu anda. Aynı görüntüyü iki yerde tutmak yerine tek sınıf var.

İKİ YAYINCI AYNI TOPIC'TE: perception (sorgu) ve pick_place (görev) aynı
MarkerArray topic'ine yazar. Bu yüzden Marker.DELETEALL KULLANILMIYOR - RViz
onu namespace'ten bağımsız uygular ve bir düğümün temizliği ötekinin
marker'larını da siler. Bunun yerine her yayıncı yalnızca KENDİ bıraktığı
id'leri hatırlar ve artık kullanmadıklarına tek tek DELETE gönderir.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

# Küre/ok/etiket üçlüsü tek bir tespit için ardışık id kullanır.
_STRIDE = 3

# Normal okunun uzunluğu (m). Emme kabının hangi yönden oturacağını göstermek
# için var; sahneyi kapatmayacak kadar kısa.
_ARROW_LENGTH = 0.12

Colour = Tuple[float, float, float]

CYAN: Colour = (0.1, 0.8, 1.0)  # sorgu / kavranacak hedef
GREEN: Colour = (0.2, 0.9, 0.3)  # bırakma hedefi
ORANGE: Colour = (1.0, 0.6, 0.1)  # yüzey normali


class DetectionMarkers:
    """
    Tespitleri küre + etiket + normal oku olarak yayınlar.

    Her örnek kendi namespace'ine yazar; aynı topic'teki başka bir örneğin
    marker'larına asla dokunmaz.
    """

    def __init__(
        self,
        node: Node,
        world_frame: str,
        namespace: str = "gemini_detections",
        topic: str = "/gemini/markers",
        lifetime_sec: float = 0.0,
        publisher: Optional[Publisher] = None,
    ) -> None:
        self._node = node
        self._frame = world_frame
        self._ns = namespace
        self._lifetime_sec = float(lifetime_sec)
        # Aynı düğümde birden çok namespace varsa tek publisher paylaşılır;
        # aynı topic'e iki publisher açmak gereksiz.
        self._pub = publisher or node.create_publisher(MarkerArray, topic, 10)
        self._live_ids: set = set()

    @property
    def publisher(self) -> Publisher:
        return self._pub

    def publish(
        self,
        detections: Sequence[Dict[str, Any]],
        colour: Colour = CYAN,
        contact: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Tespitleri çizer; önceki çizimden artakalanları siler.

        contact verilirse İLK tespitin küresi oraya konur. Bu önemsiz bir
        ayrıntı değil: bırakma noktası çarpışma yüzünden kaydırılmış olabilir
        (pick_place_node._nudge_rings) ve o durumda ER'nin ham noktasını
        çizmek, kolun GERÇEKTE gittiği yeri gizler.
        """
        array = MarkerArray()
        stamp = self._node.get_clock().now().to_msg()
        lifetime = rclpy.duration.Duration(seconds=self._lifetime_sec).to_msg()
        used: set = set()

        for index, detection in enumerate(detections):
            base = index * _STRIDE
            position = detection["position"]
            point = (position["x"], position["y"], position["z"])
            if index == 0 and contact is not None:
                point = (float(contact[0]), float(contact[1]), float(contact[2]))

            array.markers.append(
                self._sphere(base, stamp, lifetime, point, colour))
            used.add(base)

            label = detection.get("label", "?")
            surface = detection.get("surface")
            if surface is not None:
                label += f" ({surface['rms_residual'] * 1000:.1f} mm)"
            array.markers.append(
                self._text(base + 1, stamp, lifetime, point, label))
            used.add(base + 1)

            if surface is not None:
                normal = surface["normal"]
                array.markers.append(self._arrow(
                    base + 2, stamp, lifetime, point,
                    (normal["x"], normal["y"], normal["z"]),
                ))
                used.add(base + 2)

        array.markers.extend(self._deletions(used))
        self._live_ids = used
        self._pub.publish(array)

    def clear(self) -> None:
        """Bu namespace'teki her şeyi siler; başkasınınkine dokunmaz."""
        if not self._live_ids:
            return
        array = MarkerArray()
        array.markers.extend(self._deletions(set()))
        self._live_ids = set()
        self._pub.publish(array)

    # --- tek tek marker'lar -------------------------------------------------

    def _deletions(self, keep: set) -> Iterator[Marker]:
        for stale in sorted(self._live_ids - keep):
            marker = Marker()
            marker.header.frame_id = self._frame
            marker.ns = self._ns
            marker.id = stale
            marker.action = Marker.DELETE
            yield marker

    def _base(self, marker_id: int, stamp: Any, lifetime: float) -> Marker:
        marker = Marker()
        marker.header.frame_id = self._frame
        marker.header.stamp = stamp
        marker.ns = self._ns
        marker.id = marker_id
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = lifetime
        return marker

    def _sphere(
            self, marker_id: int, stamp: Any, lifetime: float, point: Sequence[float],
            colour: Sequence[float]) -> Marker:
        marker = self._base(marker_id, stamp, lifetime)
        marker.type = Marker.SPHERE
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = point
        marker.scale.x = marker.scale.y = marker.scale.z = 0.04
        marker.color.r, marker.color.g, marker.color.b = colour
        marker.color.a = 0.9
        return marker

    def _text(
            self, marker_id: int, stamp: Any, lifetime: float, point: Sequence[float],
            label: str) -> Marker:
        marker = self._base(marker_id, stamp, lifetime)
        marker.type = Marker.TEXT_VIEW_FACING
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = point[2] + 0.08
        marker.scale.z = 0.05
        marker.color.r = marker.color.g = marker.color.b = marker.color.a = 1.0
        marker.text = label
        return marker

    def _arrow(
            self, marker_id: int, stamp: Any, lifetime: float, point: Sequence[float],
            normal: Sequence[float]) -> Marker:
        marker = self._base(marker_id, stamp, lifetime)
        marker.type = Marker.ARROW
        marker.points = [
            Point(x=point[0], y=point[1], z=point[2]),
            Point(
                x=point[0] + normal[0] * _ARROW_LENGTH,
                y=point[1] + normal[1] * _ARROW_LENGTH,
                z=point[2] + normal[2] * _ARROW_LENGTH,
            ),
        ]
        marker.scale.x, marker.scale.y, marker.scale.z = 0.008, 0.016, 0.0
        marker.color.r, marker.color.g, marker.color.b = ORANGE
        marker.color.a = 0.95
        return marker
