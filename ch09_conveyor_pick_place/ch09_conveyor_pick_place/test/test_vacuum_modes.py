#!/usr/bin/env python3
"""
VGC10 mod kodlaması: 0/1/2 DEĞİL, üst bayta kaydırılmış hâlleri.

13 Ağu 2026, GERÇEK hücrede yaşandı. vacuum.py kavrama komutunu rmca=1 ile
gönderiyordu; sürücü (vgc10_control) onu reddedip hatayı bildirmek için
ROS1'den kalma `rclpy.signal_shutdown` çağırdı ve rclpy'de öyle bir şey
olmadığı için SÜRÜCÜ ÇÖKTÜ:

    AttributeError: module 'rclpy' has no attribute 'signal_shutdown'

Yani hatalı sabitin bedeli "komut gitmedi" değil, "gripper sürücüsü öldü"ydü -
üstelik ilk kavrama denemesinde, parça elde kalabilecekken.

Doğru kodlama sürücünün Modbus katmanından okunuyor:

    comModbusTcp.sendCommand:   reg_a = rmca + rvca

Yani mod ham register'ın ÜST BAYTI. %75 grip = 0x0100 + 75 = 0x014B, ki bu
OnRobotVGOutput.msg dokümanındaki örneğin aynısı.

    python3 -m pytest test/test_vacuum_modes.py -v -p no:anyio
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# vacuum.py vgc10_msgs olmadan da içe aktarılabilir (yalnız Gazebo kipi); bu
# testler yalnızca mod sabitlerini ve yapılandırmayı sınadığı için paketi istemez.

from ch09_conveyor_pick_place import vacuum  # noqa: E402

# Sürücünün kabul ettiği tek küme (baseOnRobotVG.verifyCommand).
DRIVER_ACCEPTS = (0x0000, 0x0100, 0x0200)


@pytest.mark.parametrize(
    "name", ("MODE_RELEASE", "MODE_GRIP", "MODE_IDLE")
)
def test_mode_is_one_the_driver_accepts(name):
    value = getattr(vacuum, name)
    assert value in DRIVER_ACCEPTS, (
        f"{name}={value} sürücünün kabul ettiği değerlerden biri değil "
        f"{DRIVER_ACCEPTS}; 0/1/2 gönderilirse sürücü komutu reddeder"
    )


def test_modes_are_distinct_and_shifted():
    assert vacuum.MODE_RELEASE == 0x0000
    assert vacuum.MODE_GRIP == 0x0100
    assert vacuum.MODE_IDLE == 0x0200
    # Kaydırılmamış hâller (0/1/2) sessizce geri gelmesin.
    assert vacuum.MODE_GRIP != 1 and vacuum.MODE_IDLE != 2


def test_register_matches_the_documented_example():
    """Mesaj dokümanındaki 0x014B = %75 vakumla grip örneğini toplama kuralı vermeli."""
    assert vacuum.MODE_GRIP + 75 == 0x014B
    assert vacuum.MODE_GRIP + 20 == 0x0114  # dokümandaki ikinci örnek
    assert vacuum.MODE_GRIP + 40 == 0x0128  # üçüncü örnek
    assert vacuum.MODE_IDLE + 0 == 0x0200


def test_vacuum_target_still_fits_the_low_byte():
    """
    Hedef vakum ALT bayta yazılıyor; üst sınır onu taşırmamalı.

    255'i aşan bir yüzde üst bayta taşar ve modu bozar - yani "%300 tut"
    demek sessizce BAŞKA bir mod komutu göndermek olurdu.
    """
    assert vacuum.MAX_VACUUM_PCT <= 0xFF
    # OnRobot dokümanının kendi sınırı da 80.
    assert vacuum.MAX_VACUUM_PCT == 80


def test_driver_no_longer_dies_on_a_bad_mode():
    """
    Sürücü kötü bir komutta ÖLMEMELİ; reddedip boşta bırakmalı.

    Gerçek bir hücrede sürücünün ölmesi gripper'ı olduğu durumda bırakır.
    """
    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "vgc10_control", "vgc10_control", "baseOnRobotVG.py",
    )
    if not os.path.exists(base):
        pytest.skip("vgc10_control kaynağı bu çalışma alanında yok")
    with open(base, encoding="utf-8") as handle:
        source = handle.read()
    assert "rclpy.signal_shutdown" not in source.replace(
        "# ESKİDEN rclpy.signal_shutdown", ""
    ), "rclpy.signal_shutdown geri gelmiş - rclpy'de böyle bir fonksiyon yok"


def _mode_params(name):
    import yaml
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", name,
    )
    with open(path, encoding="utf-8") as handle:
        return list(yaml.safe_load(handle).values())[0]["ros__parameters"]


def test_grip_confirmation_is_pinned_per_mode():
    """
    Vakum onayı HER İKİ mod dosyasında da AÇIKÇA yazılmalı.

    Bu ayar sim ile gerçek arasında zorunlu olarak farklı: gerçekte VGC10
    sürücüsü /OnRobotVGInput yayınlar, sim'de öyle bir sürücü yoktur ve
    kavrama Gazebo'daki DetachableJoint ile yapılır.

    14 Ağu 2026: taban dosyada tek başına true yapıldı ve SİMÜLASYONDAKİ HER
    KAVRAMA sessizce düştü ("Vakum kurulamadı"), çünkü beklenen durum mesajı
    hiç gelmiyordu. Belirti "vakum çalışmıyor"du; oysa vakum yolu sağlamdı,
    kapalı olan doğrulama kapısıydı.

    Taban dosyaya güvenmek yetmez: bir modun değeri yazılmazsa diğerinin
    ihtiyacı sessizce onu ezer. Bu yüzden test İKİ dosyada da varlığını arar.
    """
    sim = _mode_params("mode_sim.yaml")
    real = _mode_params("mode_real.yaml")

    assert "vacuum_require_confirmation" in sim, (
        "mode_sim.yaml bu ayarı yazmıyor - taban true olduğunda sim'de her "
        "kavrama başarısız sayılır"
    )
    assert sim["vacuum_require_confirmation"] is False

    assert "vacuum_require_confirmation" in real, (
        "mode_real.yaml bu ayarı yazmıyor - taban false olduğunda gerçek kol "
        "boş kavrayıcıyla yoluna devam eder"
    )
    assert real["vacuum_require_confirmation"] is True
