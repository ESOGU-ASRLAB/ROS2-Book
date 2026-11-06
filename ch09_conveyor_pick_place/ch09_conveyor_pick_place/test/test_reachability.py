#!/usr/bin/env python3
"""
ReachabilityChecker'ın karar tablosu - sahte bir IK servisiyle, MoveIt'siz.

Canlı testler (test_reachability_live.py) yalnızca hücre ayaktayken çalışıyor.
Buradaki testler her zaman çalışır ve asıl mantığı sabitler: "olmuyor"un iki
sebebi var ve ikisi FARKLI şey söyler.

    çarpışma denetimli | denetimsiz | sonuç
    -------------------|------------|---------------------------------
    SUCCESS            | -          | uygun
    başarısız          | SUCCESS    | ÇARPIŞMA (kol uzanabiliyor)
    başarısız          | başarısız  | ERİŞİM YOK (kol yetmiyor)

Ayrım önemli: çarpışmada yaklaşma mesafesini küçültmek işe yarayabilir,
erişimsizlikte yaramaz - hedefin kendisi bırakılmalıdır.

    python3 -m pytest test/test_reachability.py -v
"""

from __future__ import annotations

import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

rclpy = pytest.importorskip("rclpy")
from moveit_msgs.srv import GetPositionIK  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402

from ch09_conveyor_pick_place.reachability import (  # noqa: E402
    NO_IK_SOLUTION, SUCCESS, ReachabilityChecker,
)

QUAT = (0.0, 0.0, 0.0, 1.0)
POSITION = (0.5, 0.5, 1.0)


class FakeIKServer(Node):
    """avoid_collisions bayrağına göre ayarlanabilir yanıt veren sahte servis."""

    def __init__(self, service_name):
        super().__init__("fake_ik_server")
        self.with_collisions = SUCCESS  # avoid_collisions=True iken dönecek
        self.without_collisions = SUCCESS  # avoid_collisions=False iken dönecek
        self.calls = []
        self.seen_seed_names = None
        self.seen_link = None
        self.seen_frame = None
        self.seen_is_diff = None
        self.create_service(
            GetPositionIK, service_name, self._handle,
            callback_group=ReentrantCallbackGroup(),
        )

    def _handle(self, request, response):
        avoid = request.ik_request.avoid_collisions
        self.calls.append(avoid)
        self.seen_seed_names = list(request.ik_request.robot_state.joint_state.name)
        self.seen_link = request.ik_request.ik_link_name
        self.seen_frame = request.ik_request.pose_stamped.header.frame_id
        self.seen_is_diff = request.ik_request.robot_state.is_diff
        response.error_code.val = (
            self.with_collisions if avoid else self.without_collisions)
        return response


@pytest.fixture
def rig():
    rclpy.init()
    service_name = "/fake_compute_ik"
    server = FakeIKServer(service_name)
    client_node = Node("fake_ik_client")
    executor = MultiThreadedExecutor(4)
    executor.add_node(server)
    executor.add_node(client_node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    checker = ReachabilityChecker(
        node=client_node, group_name="g", ik_link="tool", planning_frame="world",
        service=service_name, timeout_sec=2.0,
        callback_group=ReentrantCallbackGroup(),
    )
    yield checker, server

    executor.shutdown(timeout_sec=2.0)
    spinner.join(timeout=2.0)
    server.destroy_node()
    client_node.destroy_node()
    rclpy.shutdown()


def test_valid_pose_is_accepted_with_one_call(rig):
    checker, server = rig
    result = checker.check(POSITION, QUAT)
    assert result.ok and bool(result)
    # Başarılı yolda İKİNCİ çağrı yapılmamalı: her sorgu MoveIt'e iş yükü.
    assert server.calls == [True], server.calls


def test_collision_is_distinguished_from_unreachable(rig):
    checker, server = rig
    server.with_collisions = NO_IK_SOLUTION
    server.without_collisions = SUCCESS

    result = checker.check(POSITION, QUAT)
    assert not result.ok
    assert "çarpışma" in result.reason, result.reason
    assert server.calls == [True, False], server.calls


def test_out_of_reach_is_reported_as_unreachable(rig):
    checker, server = rig
    server.with_collisions = NO_IK_SOLUTION
    server.without_collisions = NO_IK_SOLUTION

    result = checker.check(POSITION, QUAT)
    assert not result.ok
    assert "erişim yok" in result.reason, result.reason


def test_missing_service_disables_validation_instead_of_blocking(rig):
    """
    Servis yoksa hedef REDDEDİLMEZ.

    Doğrulanamayan bir hedefi reddetmek, çalışan bir görevi altyapı eksiği
    yüzünden durdurmak olurdu; amaç bilgi eklemek, kapı tutmak değil.
    """
    checker, _ = rig
    missing = ReachabilityChecker(
        node=checker._node, group_name="g", ik_link="tool",
        planning_frame="world", service="/bu_servis_yok", timeout_sec=0.5,
    )
    result = missing.check(POSITION, QUAT)
    assert result.ok
    assert "kapalı" in result.reason


def test_robot_state_is_sent_as_a_diff_so_the_payload_survives(rig):
    """
    is_diff=True olmazsa TAŞINAN PARÇA çarpışma denetiminden düşer.

    MoveIt, is_diff=False gelen bir robot_state'i sahnedekinin YERİNE koyar;
    iliştirilmiş nesneler o anda kaybolur. Doğrulama böylece planlayıcıdan
    farklı bir dünyaya bakar ve elinde kutu taşıyan kol için "uygun" der.

    ÖLÇÜLDÜ (13 Ağu 2026, canlı hücre, APPROACH_PLACE (0.924, 1.660, 1.174),
    140x83x30 mm kutu kavrayıcıda):
        is_diff=False -> IK  1  ("uygun")   <-- yanlış
        is_diff=True  -> IK -31             <-- doğru
                         temas: ur10e_stackable_bin<->gemini_payload
    Görev tam buradan patladı: kol kutuyu aldı, sonra move_group aynı poza
    plan üretemedi. Bu yüzden bayrak tohumdan BAĞIMSIZ olarak hep açık.
    """
    checker, server = rig

    assert checker.check(POSITION, QUAT).ok
    assert server.seen_is_diff is True, (
        "tohumsuz çağrıda da is_diff açık olmalı - yoksa iliştirilmiş parça düşer"
    )

    from sensor_msgs.msg import JointState

    seed = JointState()
    seed.name = ["ur10e_elbow_joint"]
    seed.position = [0.2]
    assert checker.check(POSITION, QUAT, seed=seed).ok
    assert server.seen_is_diff is True, "tohumlu çağrıda is_diff kapanmış"


def test_seed_and_link_are_forwarded_to_moveit(rig):
    """
    Tohum gönderilmezse IK, kolun bulunduğu duruştan uzak bir dala düşebilir.

    O dal geçerli çıksa bile planlayıcı oraya gitmez; yani tohumsuz bir
    doğrulama "uygun" deyip sonra hareketin başarısız olmasını seyreder.
    """
    from sensor_msgs.msg import JointState

    checker, server = rig
    seed = JointState()
    seed.name = ["ur10e_shoulder_pan_joint", "ur10e_elbow_joint"]
    seed.position = [0.1, 0.2]

    assert checker.check(POSITION, QUAT, seed=seed).ok
    assert server.seen_seed_names == seed.name
    # Doğrulama, hareketin kullandığı AYNI link ve frame üzerinden yapılmalı;
    # başka bir link için IK çözmek başka bir soruyu yanıtlamak olurdu.
    assert server.seen_link == "tool"
    assert server.seen_frame == "world"
