#!/usr/bin/env python3
"""
Planlama bütçesinin GERÇEKTEN uygulandığını doğrular.

NEDEN BU TEST VAR: pick_place_node bu iki ayarı `hasattr` ile koruyarak
yazıyordu ve yanlış isimleri deniyordu ("planning_time", "planning_attempts").
O isimler pymoveit2'de yok; hasattr sessizce False döndü, config'teki 10 s
hiçbir zaman uygulanmadı ve kütüphanenin gömülü 0.5 s'i yürürlükte kaldı.

Sonuç sahada şöyle göründü (13 Ağu 2026 koşusu): raf gözüne gitmek için
`compute_ik` tek çağrıda çarpışmasız çözüm buluyordu ama move_group
"Unable to find solution by any of the threads in 0.503596 seconds" diyordu.
Yani hata "erişilemez hedef" gibi görünüyordu, oysa sadece zaman yoktu.

Bu yüzden test isimleri MoveGroup goal alanlarına karşı sabitler. İsimler
değişirse burası kırılır - üretimde sessizce kaybolmak yerine.

    python3 -m pytest test/test_planning_budget.py -v -p no:anyio
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MoveGroup = pytest.importorskip("moveit_msgs.action").MoveGroup

# pick_place_node'un yazdığı öznitelik adları -> MoveGroup goal alan adları.
BUDGET_ATTRIBUTES = ("allowed_planning_time", "num_planning_attempts")


@pytest.mark.parametrize("attribute", BUDGET_ATTRIBUTES)
def test_move_group_goal_actually_has_the_field(attribute):
    """Yazdığımız isim MoveGroup isteğinde gerçekten bir alan olmalı."""
    assert hasattr(MoveGroup.Goal().request, attribute), (
        f"MoveGroup.Goal().request.{attribute} yok - pick_place_node "
        "bütçeyi hiçbir yere yazmıyor demektir"
    )


@pytest.mark.parametrize("module", ("book_pymoveit2",))
@pytest.mark.parametrize("attribute", BUDGET_ATTRIBUTES)
def test_pymoveit2_exposes_the_setter(module, attribute):
    """
    Kullandığımız MoveIt2 sınıfı bu ayarı setter olarak açmalı.

    hasattr koruması hâlâ kodda; amacı farklı kütüphane sürümlerini idare etmek.
    Ama koruma devreye girerse bütçe UYGULANMAZ - o yüzden burada isimlerin
    tuttuğunu ayrıca kanıtlıyoruz.
    """
    move_it2 = pytest.importorskip(module).MoveIt2
    assert isinstance(getattr(move_it2, attribute, None), property), (
        f"{module}.MoveIt2.{attribute} bir property değil"
    )
    assert getattr(move_it2, attribute).fset is not None, (
        f"{module}.MoveIt2.{attribute} salt okunur - bütçe yazılamaz"
    )


def test_node_writes_the_correct_names():
    """Kaynakta ESKİ (sessizce kaybolan) isimler kalmamalı."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ch09_conveyor_pick_place", "pick_place_node.py",
    )
    with open(path, encoding="utf-8") as handle:
        source = handle.read()

    for attribute in BUDGET_ATTRIBUTES:
        assert f'"{attribute}"' in source, f"{attribute} arm'a hiç yazılmıyor"

    # Eski isimler yalnız açıklama satırlarında geçebilir; setattr hedefi
    # olarak geçerlerse ayar yine kaybolur.
    for dead in ("self.arm.planning_time", "self.arm.planning_attempts"):
        assert dead not in source, f"{dead} pymoveit2'de yok, sessizce yutulur"
