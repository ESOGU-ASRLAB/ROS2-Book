#!/usr/bin/env python3
"""
Kavramanın iki kırılgan yeri: düzlem artığının ÖLÇÜSÜ ve sızdıran kabın telafisi.

17 Ağu 2026'da bildirilen iki belirti buraya bağlı:
  1) "yaklaşma sırasında cismin tam konumuna gitmiyor, sonra inerken cismi
     ittiriyor"  -> yamaya arka plan karışınca düzlem EĞİLİYOR; yaklaşma pozu
     temas noktasının üstünde değil YANINDA olur ve iniş nesneyi süpürür.
     Buna karşı tek savunma max_surface_rms kapısı - ve o kapı, artık yanlış
     hesaplandığı için yazılı olduğundan gevşek çalışıyordu.
  2) "kap yüzeye değiyor ama tam oturmuyor, vakum kurulamıyor"
     -> tek deneme yerine birkaç mm daha bastırıp yineleme.

    python3 -m pytest test/test_grasp_press.py -v -p no:anyio
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np = pytest.importorskip("numpy")

from ch09_conveyor_pick_place.grasp import fit_surface, is_graspable  # noqa: E402
from ch09_conveyor_pick_place.pick_place_node import GeminiPickPlaceNode  # noqa: E402


# --------------------------------------------------------------------------
# 1) Düzlem artığı RMS olmalı, mutlak sapmanın ortalaması (MAD) değil
# --------------------------------------------------------------------------


def _contaminated_patch(fraction: float, drop: float = 0.029, n: int = 440):
    """
    Kutu üstünde bir yama; noktaların `fraction` kadarı `drop` altındaki banttan.

    Gerçek durum: ER 2 kutunun kenarına yakın bir piksel gösteriyor, 12 px'lik
    yama kutunun dışına taşıyor ve derinlik bandı (5 cm) 29 mm'lik farkı
    elemiyor.
    """
    rng = np.random.default_rng(0)
    xy = rng.uniform(-0.03, 0.03, size=(n, 2))
    threshold = np.quantile(xy[:, 0], 1.0 - fraction)
    z = np.where(xy[:, 0] > threshold, -drop, 0.0)
    z = z + rng.normal(0.0, 0.0005, n)
    # fit_surface optik frame bekler: kamera orijinde, +Z ileri. Yamayı
    # kameranın 1 m önüne koy, "yükseklik" ekseni -Z olsun.
    return np.column_stack([xy[:, 0], xy[:, 1], 1.0 - z])


def test_residual_is_rms_not_mean_absolute_deviation():
    """
    sqrt(mean(d^2)) ile mean(sqrt(d^2)) aynı şey değil - ikincisi hep küçük.

    Karekök ortalamadan ÖNCE alınırsa çıkan sayı MAD olur ve max_surface_rms
    kapısı yazıldığından gevşek çalışır. Fark tam da kapının yakalaması gereken
    yerde büyür: yama iki yüzeye bölündüğünde dağılım iki tepelidir.
    """
    points = _contaminated_patch(0.06)
    patch = fit_surface(points, depth_band=0.05, min_points=25,
                        anchor=points.mean(axis=0))
    assert patch is not None

    centred = points - np.asarray(points).mean(axis=0)
    normal = np.asarray(patch.normal)
    distances = centred @ normal
    rms = float(np.sqrt((distances ** 2).mean()))
    mad = float(np.abs(distances).mean())

    assert mad < rms, "kurgu bozuk: bu yamada MAD, RMS'ten küçük olmalı"
    assert patch.rms_residual == pytest.approx(rms, rel=0.02)
    assert patch.rms_residual > mad * 1.2, (
        f"artık hâlâ MAD gibi davranıyor: {patch.rms_residual * 1000:.2f} mm "
        f"(MAD {mad * 1000:.2f}, RMS {rms * 1000:.2f})"
    )


def test_the_gate_rejects_the_patch_that_tilts_the_normal():
    """
    %6 kirlenme: MAD 4 mm kapısını geçiyordu, RMS geçmiyor.

    Geçseydi düzlem ~11 derece eğik olurdu; 15 cm'lik yaklaşmada yaklaşma pozu
    temas noktasından ~29 mm yana düşer. Bildirilen "cismi ittiriyor"
    belirtisinin geometrisi tam olarak budur.
    """
    points = _contaminated_patch(0.06)
    patch = fit_surface(points, depth_band=0.05, min_points=25,
                        anchor=points.mean(axis=0))
    assert patch is not None

    tilt = np.degrees(np.arccos(min(1.0, abs(float(np.asarray(patch.normal)[2])))))
    assert tilt > 8.0, f"kurgu bozuk: düzlem eğilmemiş ({tilt:.1f} derece)"

    ok, reason = is_graspable(
        patch, min_inliers=40, max_rms=0.004, max_tilt_deg=180.0)
    assert not ok
    assert "düz değil" in reason, reason


def test_a_clean_patch_still_passes():
    """Kapı sıkılaştı ama TEMİZ yamayı elemiyor - yoksa hiçbir şey kavranamaz."""
    rng = np.random.default_rng(1)
    xy = rng.uniform(-0.03, 0.03, size=(440, 2))
    points = np.column_stack([xy[:, 0], xy[:, 1],
                              1.0 + rng.normal(0.0, 0.0005, 440)])
    patch = fit_surface(points, depth_band=0.05, min_points=25,
                        anchor=points.mean(axis=0))
    assert patch is not None
    ok, reason = is_graspable(
        patch, min_inliers=40, max_rms=0.004, max_tilt_deg=180.0)
    assert ok, reason


# --------------------------------------------------------------------------
# 2) Sızdıran kabı birkaç mm daha bastırıp yineleme
# --------------------------------------------------------------------------


class _Logger:
    def __init__(self):
        self.lines = []

    def info(self, message):
        self.lines.append(message)

    def warning(self, message):
        self.lines.append(message)


class PressStub:
    """_grip_with_press'in dokunduğu her şey - düğüm ayağa kaldırmadan."""

    def __init__(self, grip_succeeds_at=0, move_fails_at=None, retries=2):
        self.touch_offset = -0.008
        self.touch_press_step = 0.004
        self.touch_press_retries = retries
        self._grip_calls = 0
        self._grip_succeeds_at = grip_succeeds_at
        self._move_fails_at = move_fails_at
        self.moves = []
        self._logger = _Logger()

    def get_logger(self):
        return self._logger

    def _grip(self):
        result = self._grip_calls == self._grip_succeeds_at
        self._grip_calls += 1
        return result

    def _move_pose(self, position, quat, label, cartesian=False):
        self.moves.append((label, position, cartesian))
        if self._move_fails_at is not None and len(self.moves) >= self._move_fails_at:
            return False
        return True

    _offset = staticmethod(GeminiPickPlaceNode._offset)
    grip_with_press = GeminiPickPlaceNode._grip_with_press


CONTACT = (0.60, 0.10, 0.845)
UP = (0.0, 0.0, 1.0)
QUAT = (1.0, 0.0, 0.0, 0.0)


def test_no_extra_press_when_the_first_attempt_holds():
    """İyi durumda hiç fazladan basma olmamalı - parçayı bantta ittirirdi."""
    stub = PressStub(grip_succeeds_at=0)
    assert stub.grip_with_press(CONTACT, UP, QUAT) is True
    assert stub.moves == []


def test_press_goes_deeper_along_the_normal_only():
    """Basma NORMAL boyunca: yanal bileşeni olsa parçayı kaydırırdı."""
    stub = PressStub(grip_succeeds_at=1)
    assert stub.grip_with_press(CONTACT, UP, QUAT) is True

    assert len(stub.moves) == 1
    label, position, cartesian = stub.moves[0]
    assert label == "TOUCH_PRESS_1"
    assert cartesian is True, "serbest plan parçanın yanından girebilir"
    assert position[0] == pytest.approx(CONTACT[0], abs=1e-9)
    assert position[1] == pytest.approx(CONTACT[1], abs=1e-9)
    # -0.008 (touch_offset) - 0.004 = 12 mm bastırma
    assert position[2] == pytest.approx(CONTACT[2] - 0.012, abs=1e-9)


def test_press_is_bounded_by_the_retry_count():
    """Toplam basma step * retries'i AŞMAMALI - kauçuğun ezilme payı sınırlı."""
    stub = PressStub(grip_succeeds_at=99)  # hiç tutmaz
    assert stub.grip_with_press(CONTACT, UP, QUAT) is False

    assert [label for label, _, _ in stub.moves] == [
        "TOUCH_PRESS_1", "TOUCH_PRESS_2"]
    deepest = min(position[2] for _, position, _ in stub.moves)
    assert deepest == pytest.approx(CONTACT[2] - 0.016, abs=1e-9)


def test_retries_can_be_switched_off():
    """touch_press_retries: 0 eski davranışa döner: tek deneme, hareket yok."""
    stub = PressStub(grip_succeeds_at=99, retries=0)
    assert stub.grip_with_press(CONTACT, UP, QUAT) is False
    assert stub.moves == []


def test_unplannable_press_gives_up_instead_of_retrying_blindly():
    """Derine inilemiyorsa vakumu tekrar denemenin anlamı yok - kap yerinde."""
    stub = PressStub(grip_succeeds_at=99, move_fails_at=1)
    assert stub.grip_with_press(CONTACT, UP, QUAT) is False
    assert len(stub.moves) == 1


# --------------------------------------------------------------------------
# 3) Ölçülen parça kutusu gerçekten KULLANILMALI
# --------------------------------------------------------------------------


MEASURED = {
    "size": [0.153, 0.094, 0.036],
    "size_measured": [0.143, 0.084, 0.026],
    "centre": [0.648, 0.059, 0.861],
    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    "pixels": 4968,
}


class BoxStub:
    """_payload_box'ın dokunduğu her şey."""

    payload_size = (0.140, 0.083, 0.030)
    # QUAT = (1,0,0,0) ile bu eksen dünya -Z'ye eşlenir, yani normal +Z olur:
    # config yolundaki merkez hesabının anlamlı çıkması için gerekli.
    tool_approach_vector = (0.0, 0.0, 1.0)

    _payload_box = GeminiPickPlaceNode._payload_box
    _offset = staticmethod(GeminiPickPlaceNode._offset)
    _payload_normal = GeminiPickPlaceNode._payload_normal


def test_measured_box_is_read_from_the_surface_patch():
    """
    Konumlandırıcı (locator) ölçümü detection["surface"]["payload"] içine yazar.

    17 Ağu 2026 koşusunda ölçüm başarılıydı (143x84x26) ama iliştirilen kutu
    config'ten geldi (140x83x30): okuyan taraf bir katman yukarıya bakıyordu.
    """
    detection = {"surface": {"payload": dict(MEASURED)}}
    size, centre, quat, source = BoxStub()._payload_box(
        detection, CONTACT, QUAT)

    assert source == "ölçüldü", "ölçüm yine atılıyor, config'e düşülüyor"
    assert tuple(size) == (0.153, 0.094, 0.036)
    assert tuple(centre) == (0.648, 0.059, 0.861)
    assert tuple(quat) == (0.0, 0.0, 0.0, 1.0)


def test_flat_layout_still_works():
    """Ölçüm tespitin kökündeyse de bulunmalı - iki yapı da desteklenir."""
    detection = {"payload": dict(MEASURED)}
    _, _, _, source = BoxStub()._payload_box(detection, CONTACT, QUAT)
    assert source == "ölçüldü"


def test_config_fallback_only_when_there_is_no_measurement():
    """Ölçüm YOKSA config kutusu; kaynak da öyle etiketlenmeli."""
    detection = {"surface": {}}
    size, _, _, source = BoxStub()._payload_box(detection, CONTACT, QUAT)
    assert source == "config (ölçülemedi)"
    assert tuple(size) == BoxStub.payload_size
