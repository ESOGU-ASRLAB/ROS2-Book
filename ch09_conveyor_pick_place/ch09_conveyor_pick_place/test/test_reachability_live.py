#!/usr/bin/env python3
"""
Hedef doğrulamasının CANLI sahnede doğru kararı verdiğini gösterir.

Çalışan bir move_group gerektirir; yoksa atlanır (skip). Pozlar 11 Ağu 2026'da
başarısız olan gerçek koşudan alındı - yani bu test, o koşuyu tekrar etseydik
yeni kodun ne yapacağını sorar:

  * ER'nin seçtiği göz (üstten 2. sıra, tavanlı)  -> REDDEDİLMELİ
  * toolkit üst sırası (yukarı açık)               -> KABUL EDİLMELİ
  * konveyördeki kutunun yaklaşma pozu             -> KABUL EDİLMELİ

    python3 -m pytest test/test_reachability_live.py -v
"""

from __future__ import annotations

import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

rclpy = pytest.importorskip("rclpy")
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402

from ch09_conveyor_pick_place.grasp import approach_quaternion  # noqa: E402
from ch09_conveyor_pick_place.reachability import ReachabilityChecker  # noqa: E402

GROUP = "real_ur10e"
EE_LINK = "ur10e_suction_cup"
TOOL_AXIS = (0.8660254, 0.5, 0.0)

# Yatay yüzey (normal +Z) için kavrama oryantasyonu - koşudaki ile aynı
QUAT = approach_quaternion((0.0, 0.0, 1.0), approach_vector=TOOL_AXIS)


@pytest.fixture(scope="module")
def checker():
    rclpy.init()
    node = Node("reachability_live_test")
    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    obj = ReachabilityChecker(
        node=node, group_name=GROUP, ik_link=EE_LINK,
        planning_frame="world", timeout_sec=3.0,
        callback_group=ReentrantCallbackGroup(),
    )
    if not obj.available():
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        rclpy.shutdown()
        pytest.skip("/compute_ik yok - move_group çalışmıyor")

    yield obj

    executor.shutdown(timeout_sec=2.0)
    spinner.join(timeout=2.0)
    node.destroy_node()
    rclpy.shutdown()


def test_ceilinged_compartment_is_rejected(checker):
    """Başarısız koşunun APPROACH_PLACE frame pozu: kap rafın içinden geçiyordu."""
    result = checker.check((0.735, 1.636, 0.963), QUAT)
    assert not result.ok, "tavanlı göz kabul edilmemeliydi"
    # Erişim sorunu DEĞİL, çarpışma: kol oraya uzanabiliyor.
    assert "çarpışma" in result.reason, result.reason


def test_top_row_is_accepted(checker):
    """Toolkit üst sırası (taban 0.938) yukarı açık; oraya bırakılabilmeli."""
    assert checker.check((0.735, 1.636, 1.0953), QUAT).ok


def test_pick_approach_is_accepted(checker):
    """Konveyördeki kutunun yaklaşma pozu - koşuda zaten çalışmıştı."""
    assert checker.check((0.749, 0.101, 1.092), QUAT).ok


def test_far_out_of_reach_is_reported_as_unreachable(checker):
    """Çarpışma ile erişimsizlik AYRI raporlanmalı: teşhis için gerekli."""
    result = checker.check((5.0, 5.0, 2.5), QUAT)
    assert not result.ok
    assert "erişim yok" in result.reason, result.reason


def test_shrinking_the_approach_rescues_the_top_row(checker):
    """
    Üst sıra gözüne EN AZ BİR yaklaşma mesafesinden gidilebilmeli.

    Aday listesinin varlık sebebi: tek sabit mesafe iki farklı yere birden
    uymuyor. Hangi mesafenin tutacağı sahneye bağlıdır ve zamanla değişir -
    11 Ağu 2026'da 15 cm çarpışıyordu, aynı gün konveyör hizalandıktan sonra
    hepsi geçerli çıktı.

    Bu yüzden test BELİRLİ bir mesafenin başarısız olmasını ŞART KOŞMAZ.
    Öyle yazılmıştı ve sahne düzelince test kırıldı - yani sistemin
    iyileşmesini hata gibi raporladı. Ölçtüğü şey artık şu: hedefe
    gidilebiliyor ve küçültmek işi bozmuyor.
    """
    floor_z = 0.938
    results = {}
    for distance in (0.15, 0.12, 0.10, 0.08):
        frame_z = floor_z + distance + 0.0843  # uç -> frame (TCP 84.3 mm)
        results[distance] = checker.check((0.735, 1.636, frame_z), QUAT).ok

    assert any(results.values()), \
        f"üst sıraya hiçbir mesafeden gidilemiyor: {results}"


def test_sideways_place_normal_is_the_actual_failure(checker):
    """
    Bırakma normali YATAY olunca hedef ulaşılamaz görünüyor - asıl hata bu.

    11 Ağu 2026 koşusu: ER üst bölmeyi doğru gösterdi (0.884, 1.688, 0.963)
    ama işaret edilen piksel rafın ön yüzüne düştüğü için normal (0,-1,0)
    çıktı. Kol göze yukarıdan inmek yerine yandan girmeye çalıştı ve beş
    yaklaşma mesafesinin beşi de çarpıştı. Aynı noktaya yukarıdan inilebiliyor.

    _grasp_frame(for_grasp=False) bu yüzden yatay normali +Z ile değiştiriyor.
    """
    contact = (0.884, 1.688, 0.963)
    q_side = approach_quaternion((0.0, -1.0, 0.0), approach_vector=TOOL_AXIS)

    sideways = any(
        checker.check(
            (contact[0], contact[1] - d, contact[2] + 0.0843), q_side
        ).ok
        for d in (0.15, 0.12, 0.10, 0.08, 0.06)
    )
    assert not sideways, "yandan giriş geçerli çıktıysa teşhis artık geçerli değil"

    from_above = any(
        checker.check(
            (contact[0], contact[1], contact[2] + d + 0.0843), QUAT
        ).ok
        for d in (0.15, 0.12, 0.10, 0.08, 0.06)
    )
    assert from_above, "yukarıdan iniş de olmuyorsa sorun normalde değil"


def test_lower_rows_are_all_rejected(checker):
    """
    Alt sıralar bu gripper ile kullanılamaz; doğrulama onları ELEMELİ.

    Görev başarısızlığının teşhisi buydu: ER üst sıra yerine alt gözleri
    gösteriyordu ve hepsi 'çarpışma' ile geri döndü. Üst sıra ise aynı anda
    geçerliydi (bkz. test_top_row_is_accepted) - yani sorun geometri değil,
    ER'nin gösterdiği nokta.
    """
    rejected = {}
    for floor_z in (0.259, 0.517, 0.727):  # ölçülen raf tabanları, üst sıra hariç
        frame_z = floor_z + 0.06 + 0.0843
        rejected[floor_z] = checker.check((0.735, 1.636, frame_z), QUAT).ok
    assert not any(rejected.values()), \
        f"tavanlı bir göz kabul edildi: {rejected}"
