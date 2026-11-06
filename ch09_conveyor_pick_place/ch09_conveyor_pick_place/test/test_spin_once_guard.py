#!/usr/bin/env python3
"""
rclpy.spin_once koruması: düğüm sahibini asla değiştirmemeli.

NEDEN: pymoveit2 future beklerken rclpy.spin_once(self._node) çağırıyor, o da
düğümü global bir SingleThreadedExecutor'a alıyor. Bir düğüm tek bir executor'a
ait olabildiği için bu, düğümü bizim MultiThreadedExecutor'ımızdan söküyor -
biz spin ederken. Sonuç, 11 Ağu 2026'da ölçülen çökme:

    RCLError: Failed to get number of ready entities for action client:
    wait set index for status subscription is out of bounds

Bu testler yamanın çalıştığını DOĞRUDAN gösteriyor: yamasız hâlde düğüm
executor'ından düşüyor, yamalı hâlde düşmüyor.

    python3 -m pytest test/test_spin_once_guard.py -v
"""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

rclpy = pytest.importorskip("rclpy")
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402

from ch09_conveyor_pick_place.pick_place_node import install_spin_once_guard  # noqa: E402


@pytest.fixture
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def restore_spin_once():
    original = rclpy.spin_once
    yield
    rclpy.spin_once = original


def test_unguarded_spin_once_steals_the_node(ros):
    """
    Yamasız davranışı sabitler: bu testin geçmesi hatanın GERÇEK olduğunu gösterir.

    ÖLÇÜLEN davranış (Humble): spin_once'ın finally'sindeki remove_node yalnızca
    global executor'ın _nodes kümesinden siliyor, düğümün KENDİ executor
    referansına dokunmuyor. Yani düğüm çağrıdan sonra:
      * bizim executor'ımızın _nodes'unda DEĞİL (add_node onu oradan söktü),
      * global executor'ın _nodes'unda da DEĞİL (finally sildi),
      * ama node.executor hâlâ GLOBAL executor'ı gösteriyor.
    Sahiplik el değiştirmiş oluyor; wait set'i bozan da bu.
    """
    node = Node("guard_test_unpatched")
    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    assert node.executor is executor

    rclpy.spin_once(node, timeout_sec=0.05)

    assert node.executor is not executor, "yamasız spin_once sahipliği almalıydı"
    assert node not in list(executor.get_nodes())
    node.destroy_node()


def test_guarded_spin_once_keeps_the_owner(ros, restore_spin_once):
    node = Node("guard_test_patched")
    executor = MultiThreadedExecutor(2)
    executor.add_node(node)

    install_spin_once_guard()
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=1.0)

    assert node.executor is executor, "yamalı spin_once sahibi değiştirmemeli"
    node.destroy_node()


def test_guarded_spin_once_still_works_for_ownerless_nodes(ros, restore_spin_once):
    """Sahipsiz düğümde ESKİ davranış korunmalı; yama her şeyi uykuya çevirmemeli."""
    install_spin_once_guard()
    node = Node("guard_test_ownerless")
    assert node.executor is None

    ticks = []
    node.create_timer(0.01, lambda: ticks.append(1))
    deadline = time.time() + 2.0
    while not ticks and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    assert ticks, "sahipsiz düğümde spin_once gerçekten spin etmeli"
    node.destroy_node()


def test_callbacks_still_run_while_guarded_spin_once_waits(ros, restore_spin_once):
    """
    Asıl soru: yama varken future'ların hâlâ tamamlandığını sınar.

    pymoveit2'nin döngüsü `while not future.done(): rclpy.spin_once(node)`.
    Yama uyuduğuna göre ilerlemeyi BİZİM executor'ımız sağlamalı - yoksa
    çökme yerine kilitlenme almış oluruz ki daha kötüdür.
    """
    node = Node("guard_test_progress")
    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    install_spin_once_guard()

    ticks = []
    node.create_timer(0.02, lambda: ticks.append(1))

    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        # pymoveit2'nin bekleme döngüsünü birebir taklit et
        deadline = time.time() + 3.0
        while not ticks and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=1.0)
    finally:
        executor.shutdown(timeout_sec=2.0)
        spinner.join(timeout=2.0)

    assert ticks, "executor spin ederken timer'ın tetiklenmesi gerekirdi"
    assert node.executor is executor
    node.destroy_node()
