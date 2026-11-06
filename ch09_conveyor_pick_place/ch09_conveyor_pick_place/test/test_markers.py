#!/usr/bin/env python3
"""
RViz marker'larının paylaşılan mantığı - ROS grafiği ayağa kaldırmadan.

NEDEN VAR: bu kod eskiden perception_node'un içindeydi ve DELETEALL
kullanıyordu. İki düğüm aynı topic'e yazmaya başlayınca DELETEALL sessiz bir
silah oluyor: RViz onu namespace'ten bağımsız uygular, yani bir sorgu görevin
marker'larını süpürür. Belirti "marker'lar bazen kayboluyor" olur ve
tekrarlanması zordur.

    python3 -m pytest test/test_markers.py -v -p no:anyio
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("visualization_msgs")

from visualization_msgs.msg import Marker  # noqa: E402

from ch09_conveyor_pick_place.markers import CYAN, GREEN, DetectionMarkers  # noqa: E402


class FakeClock:
    def now(self):
        class Stamp:
            def to_msg(self):
                from builtin_interfaces.msg import Time
                return Time()
        return Stamp()


class FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, array):
        self.published.append(array)


class FakeNode:
    def __init__(self):
        self.publisher = FakePublisher()

    def create_publisher(self, *args, **kwargs):
        return self.publisher

    def get_clock(self):
        return FakeClock()


def detection(label="kutu", x=1.0, y=2.0, z=0.5, with_surface=True):
    item = {"label": label, "position": {"x": x, "y": y, "z": z}}
    if with_surface:
        item["surface"] = {
            "centroid": {"x": x, "y": y, "z": z},
            "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
            "rms_residual": 0.0012,
            "inliers": 400,
            "extent": 0.1,
        }
    return item


@pytest.fixture
def rig():
    node = FakeNode()
    return node, DetectionMarkers(node, world_frame="world", namespace="test_ns")


def test_deleteall_is_never_used(rig):
    """
    DELETEALL yasak: aynı topic'teki DİĞER düğümün çizimini de siler.

    RViz'in marker display'i DELETEALL'ı namespace'e bakmadan uygular. Sorgu
    ile görev aynı topic'i paylaştığı için bu, birinin ötekini süpürmesi
    demektir.
    """
    node, markers = rig
    markers.publish([detection()])
    markers.publish([])
    markers.clear()

    for array in node.publisher.published:
        for marker in array.markers:
            assert marker.action != Marker.DELETEALL, "DELETEALL geri gelmiş"


def test_stale_markers_are_deleted_one_by_one(rig):
    """
    Tespit sayısı azaldığında artakalanlar TEK TEK silinmeli.

    Silinmezse eski tespitler RViz'de asılı kalır ve bu koşuya aitmiş gibi
    okunur - yanlış yerde hata aranmasına yol açar.
    """
    node, markers = rig
    markers.publish([detection("a"), detection("b", y=2.5)])
    first = node.publisher.published[-1]
    assert len([m for m in first.markers if m.action == Marker.ADD]) == 6

    markers.publish([detection("a")])
    second = node.publisher.published[-1]
    deleted = sorted(m.id for m in second.markers if m.action == Marker.DELETE)
    assert deleted == [3, 4, 5], deleted


def test_clear_only_touches_its_own_namespace(rig):
    node, markers = rig
    markers.publish([detection()])
    markers.clear()
    for marker in node.publisher.published[-1].markers:
        assert marker.ns == "test_ns"
        assert marker.action == Marker.DELETE


def test_two_namespaces_can_share_one_publisher():
    """Kavrama ve bırakma aynı publisher'ı paylaşır ama birbirini silmez."""
    node = FakeNode()
    pick = DetectionMarkers(node, "world", namespace="gemini_pick")
    place = DetectionMarkers(
        node, "world", namespace="gemini_place", publisher=pick.publisher)

    pick.publish([detection("kutu")], colour=CYAN)
    place.publish([detection("göz", y=1.6)], colour=GREEN)

    # place'in yayını, pick'in id'lerine DELETE göndermemeli.
    last = node.publisher.published[-1]
    assert all(m.ns == "gemini_place" for m in last.markers)
    assert not any(m.action == Marker.DELETE for m in last.markers)


def test_contact_override_moves_the_sphere_not_the_raw_point(rig):
    """
    Kaydırılmış temas noktası çizilmeli, ER'nin ham noktası değil.

    13 Ağu 2026: bırakma noktası çarpışma yüzünden 54 mm kaydırıldı. Ham
    noktayı çizmek, kolun GERÇEKTE gittiği yeri gizler - yani görselleştirme
    tam da açıklaması gereken farkı saklar.
    """
    node, markers = rig
    markers.publish([detection(x=0.925, y=1.659)], contact=(0.925, 1.605, 0.94))

    sphere = next(
        m for m in node.publisher.published[-1].markers
        if m.type == Marker.SPHERE
    )
    assert sphere.pose.position.y == pytest.approx(1.605)


def test_arrow_is_skipped_when_there_is_no_surface(rig):
    """Yüzey yoksa normal oku ÇİZİLMEMELİ - uydurma bir yön göstermek yanlış."""
    node, markers = rig
    markers.publish([detection(with_surface=False)])
    types = [m.type for m in node.publisher.published[-1].markers]
    assert Marker.ARROW not in types
    assert Marker.SPHERE in types


def test_arrow_starts_at_the_contact_point(rig):
    """Ok temas noktasından çıkmalı; kaydırma varsa ok da kaymalı."""
    node, markers = rig
    markers.publish([detection(x=1.0, y=2.0, z=0.5)], contact=(1.0, 1.9, 0.5))
    arrow = next(
        m for m in node.publisher.published[-1].markers if m.type == Marker.ARROW
    )
    assert arrow.points[0].y == pytest.approx(1.9)
    # Uç, normal (+Z) boyunca yukarıda olmalı.
    assert arrow.points[1].z > arrow.points[0].z
