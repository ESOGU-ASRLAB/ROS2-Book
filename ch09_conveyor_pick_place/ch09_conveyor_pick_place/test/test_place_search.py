#!/usr/bin/env python3
"""
Bırakma noktasının yüzey boyunca kaydırılarak aranması.

NEDEN VAR: ER gözü doğru buluyor ama noktayı ortasına koymuyor. 13 Ağu 2026'da
üst göz için (0.925, 1.659) döndü; gözün y sınırları 1.519/1.691, ortası 1.605.
83 mm genişliğindeki kutu kabın altında ortalandığı için 1.7005'e taşıyor ve
bölme duvarının 9.5 mm içine giriyor.

Buradaki testler MoveIt'siz çalışır ve üç şeyi sabitler:
  * kaydırma YÜZEY BOYUNCA olur (normale dik), asla yüzeyin içine/üstüne
  * içten dışa aranır - en yakın geçerli nokta seçilir
  * KAVRAMADA kaydırma YOKTUR (kap parçanın yanına inerdi)

    python3 -m pytest test/test_place_search.py -v -p no:anyio
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np = pytest.importorskip("numpy")

from ch09_conveyor_pick_place.pick_place_node import GeminiPickPlaceNode  # noqa: E402


class Stub:
    """_nudge_rings'in ihtiyaç duyduğu iki alan - düğüm ayağa kaldırmadan."""

    def __init__(self, radii=(0.02, 0.04, 0.06), directions=8):
        self.place_search_radii = sorted(radii)
        self.place_search_directions = directions

    rings = GeminiPickPlaceNode._nudge_rings

    def nudged(self, contact, normal):
        """Halkaları düzleştirir - eski testler nokta bazında yazılmıştı."""
        for radius, ring in self.rings(contact, normal):
            for point in ring:
                yield point, radius


CONTACT = (0.925, 1.659, 0.940)
UP = (0.0, 0.0, 1.0)


def test_nudges_stay_on_the_surface_plane():
    """Kaydırma normale DİK olmalı: z değişirse yaklaşma mesafesi bozulur."""
    for point, _ in Stub().nudged(CONTACT, UP):
        assert point[2] == pytest.approx(CONTACT[2], abs=1e-9), point


def test_nudges_have_the_requested_radius():
    for point, radius in Stub().nudged(CONTACT, UP):
        distance = math.hypot(point[0] - CONTACT[0], point[1] - CONTACT[1])
        assert distance == pytest.approx(radius, abs=1e-9)


def test_search_goes_inside_out():
    """Yarıçaplar artan sırada gelmeli - yoksa uzak bir göze savrulabiliriz."""
    radii = [radius for _, radius in Stub().nudged(CONTACT, UP)]
    assert radii == sorted(radii), radii
    assert radii[0] == 0.02 and radii[-1] == 0.06


def test_the_measured_failure_is_within_reach_of_the_search():
    """
    Gerçek kaçıklık (54 mm) arama yarıçapının içinde kalmalı.

    Aksi halde arama var ama yetişmiyor demektir - bu testin tek işi, en büyük
    yarıçapın ölçülen hatadan küçük yazılmasını engellemek.
    """
    centre_y = (1.519 + 1.691) / 2.0  # ölçülen bölme sınırları
    error = abs(CONTACT[1] - centre_y)
    assert error == pytest.approx(0.054, abs=0.001)
    assert max(Stub().place_search_radii) >= error, (
        "arama yarıçapı ER'nin ölçülen kaçıklığından küçük"
    )

    # Kutu (83 mm geniş) ortalandığında duvarların içinde kalmalı; aramanın
    # bulması GEREKEN noktanın var olduğunu burada kanıtlıyoruz.
    assert 1.519 < centre_y - 0.0415 and centre_y + 0.0415 < 1.691


def test_at_least_one_nudge_lands_near_the_compartment_centre():
    """Aday kümesi gerçekten ortaya yakın bir nokta ÜRETMELİ."""
    centre_y = (1.519 + 1.691) / 2.0
    best = min(
        abs(point[1] - centre_y) for point, _ in Stub().nudged(CONTACT, UP)
    )
    # 8 yön x 6 cm ile ortadan 2 cm'den fazla sapmamalıyız.
    assert best < 0.02, f"en iyi aday ortadan {best * 1000:.0f} mm uzakta"


def test_tilted_surface_nudges_stay_in_that_surface():
    """Eğik bir yüzeyde kaydırma O YÜZEYİN düzleminde kalmalı."""
    normal = (0.0, math.sin(math.radians(30)), math.cos(math.radians(30)))
    unit = np.asarray(normal) / np.linalg.norm(normal)
    for point, _ in Stub().nudged(CONTACT, normal):
        offset = np.asarray(point) - np.asarray(CONTACT)
        assert float(offset @ unit) == pytest.approx(0.0, abs=1e-9)


def test_empty_radii_disables_the_search():
    assert list(Stub(radii=()).nudged(CONTACT, UP)) == []


def test_degenerate_normal_is_ignored_instead_of_crashing():
    """Sıfır normal ölçüm artefaktıdır; arama sessizce boş dönmeli."""
    assert list(Stub().nudged(CONTACT, (0.0, 0.0, 0.0))) == []


def test_grasp_targets_are_never_nudged():
    """
    Kavrama noktası kutsal: kaydırılırsa kap parçanın YANINA iner.

    Kaynak düzeyinde sabitliyoruz - arama çağrısı `need_grasp` korumasının
    içinde kalmalı.
    """
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ch09_conveyor_pick_place", "pick_place_node.py",
    )
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()

    call = next(
        index for index, line in enumerate(lines)
        if "self._nudge_rings(contact, normal)" in line
    )
    guard = next(
        index for index in range(call, 0, -1)
        if lines[index].strip() == "if not need_grasp:"
    )
    # Koruma ile çağrı arasında başka bir blok açılmamalı.
    assert 0 < call - guard <= 3, (
        "_nudge_rings çağrısı 'if not need_grasp' korumasının altında değil"
    )


def test_distance_loop_is_outside_the_direction_loop():
    """
    Mesafe DIŞ, yön İÇ döngüde olmalı - hem hız hem tercih sırası için.

    Aynı x/y'de yaklaşmayı kısaltmak YANAL bir sığmama sorununu çözmez, çünkü
    kutu yerinde duruyor. Yön dışta gezildiğinde her aday nokta için beş
    mesafe de boşuna deneniyordu; ölçülen koşuda arama 185 s sürdü.

    Ayrıca approach_candidates büyükten küçüğe sıralı: bu sıra BÜYÜK mesafe
    tercihini korur. Ters çevrilirse 2. yöndeki 15 cm yerine 1. yöndeki 6 cm
    kabul edilir - yani kutu gereksizce gözün derinine sokulur.
    """
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ch09_conveyor_pick_place", "pick_place_node.py",
    )
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()

    ring = next(
        index for index, line in enumerate(lines)
        if "self._nudge_rings(contact, normal)" in line
    )
    distance_loop = next(
        index for index in range(ring, ring + 30)
        if lines[index].strip() == "for distance in candidates:"
    )
    direction_loop = next(
        index for index in range(ring, ring + 30)
        if lines[index].strip() == "for shifted in ring:"
    )
    assert distance_loop < direction_loop, (
        "yön döngüsü mesafe döngüsünün dışına çıkmış - arama yavaşlar ve "
        "gereksizce küçük yaklaşma mesafesi seçilir"
    )


def test_place_approach_starts_above_the_structure():
    """
    BIRAKMA yaklaşma listesi BÜYÜKTEN başlamalı ve yeterince büyük olmalı.

    Doğrulama bir DURUM denetler (IK); bu arıza ise YOLDAYDI. 13 Ağu 2026,
    nokta gözün ortasına çekildikten sonra ölçüldü:

        d = 20 cm -> kutu altı 1.109 -> plan BAŞARILI
        d = 18 cm -> kutu altı 1.089 -> plan BAŞARISIZ (-2)

    Sınır rafın üst kenarı 1.095. Kutunun altı bunu geçmezse parça hedefte
    zaten gözün içindedir ve planlayıcı onu YANDAN sokmaya çalışır; IK ise
    her mesafede "geçerli" der, yani doğrulama bunu göremez.

    Kavrama listesi küçükten gider (dar yere sokulmak için); bu ikisinin
    karışması sessizce eski hatayı geri getirir.
    """
    import yaml

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "gemini_params.yaml",
    )
    with open(path, encoding="utf-8") as handle:
        params = list(yaml.safe_load(handle).values())[0]["ros__parameters"]

    place = [float(v) for v in params["place_approach_candidates"]]
    assert place == sorted(place, reverse=True), (
        f"bırakma listesi büyükten küçüğe olmalı: {place}"
    )

    # Ölçülen eşik: raf kenarı 1.095, göz tabanı 0.939, kutu yüksekliği 30 mm.
    needed = (1.095 - 0.939) + 0.030
    assert place[0] >= needed, (
        f"ilk bırakma mesafesi {place[0] * 100:.0f} cm, yapının üstünü geçmek "
        f"için {needed * 100:.0f} cm gerekiyor"
    )

    # Parça 25 cm'den DÜŞMEMELİ: yaklaşma yükseldiği için iniş şart oldu.
    assert params["descend_before_release"] is True, (
        "yaklaşma yapının üstüne çıktı; iniş kapalıysa parça o yükseklikten düşer"
    )


def test_release_pose_is_validated_and_a_short_descent_is_not_fatal():
    """
    İki ayrı ders, ikisi de 13 Ağu 2026 koşusundan.

    1) Yaklaşma yapının üstüne çıkınca ER'nin ORTALANMAMIŞ noktası (0.922,
       1.650) yaklaşma denetimini GEÇTİ - kutu o yükseklikte kenarın
       üzerindeydi - ve kaydırma hiç tetiklenmedi. Kusur bir aşama sonraya
       taşındı: dik iniş, kutu kenar hizasına gelince durdu (fraction 0.293),
       çünkü 1.650 + 0.0415 = 1.6915 > 1.691 (bölme duvarı).
       Bu yüzden arama İNİŞ POZUNU da denetlemeli.

    2) O iniş yarıda kaldığında görev ölmemeli: ulaşılan nokta bırakmak için
       uygundu. Eşik bu yüzden TOUCH'ınkinden ayrı ve gevşek.
    """
    import yaml

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "ch09_conveyor_pick_place", "pick_place_node.py"),
              encoding="utf-8") as handle:
        source = handle.read()

    # (1) iniş denetimi hem taban noktada hem halkada uygulanmalı
    assert source.count("self._release_ok(") >= 2, (
        "iniş denetimi hem taban noktada hem halka elemesinde çağrılmalı"
    )
    assert "if need_grasp or self._release_ok(" in source, (
        "iniş denetimi KAVRAMA yolunu engellememeli, yalnız bırakmada çalışmalı"
    )

    # (2) iniş başarısızlığı görevi düşürmemeli
    assert '"Bırakma pozuna inilemedi"' not in source, (
        "yarım kalan iniş hâlâ görevi öldürüyor"
    )
    assert "min_fraction=self.release_min_fraction" in source

    with open(os.path.join(root, "config", "gemini_params.yaml"),
              encoding="utf-8") as handle:
        params = list(yaml.safe_load(handle).values())[0]["ros__parameters"]

    assert params["release_min_fraction"] < params["cartesian_min_fraction"], (
        "iniş eşiği TOUCH eşiğinden gevşek olmalı"
    )
    # TOUCH gevşerse kap parçaya değmeden kavrar - bu sessiz ve pahalı bir hata.
    assert params["cartesian_min_fraction"] >= 0.95
