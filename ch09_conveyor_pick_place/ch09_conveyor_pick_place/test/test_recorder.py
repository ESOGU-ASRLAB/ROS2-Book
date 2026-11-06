#!/usr/bin/env python3
"""
Kayıt yolunun ROS'suz sınanabilen parçaları.

Kayıt düğümünün kendisi bir ROS grafiği ister; burada sınanan, o düğümün
DOĞRULUĞUNUN dayandığı iki sessiz varsayım:

  1. Ölçüm veri yolu, JSON'a çevrilemeyen bir değer geldiğinde sessizce
     bozulmamalı. numpy skalerleri ve NaN, düzlem oturtmadan doğrudan
     geliyor; ilki json.dumps'ı patlatır, ikincisi standart dışı `NaN`
     üretip satırı okunamaz hâle getirir.
  2. CSV yazıcı her satırda diske inmeli. Koşu Ctrl-C ile kesildiğinde
     tamponda kalan satır, tam da en çok merak edilen son satırdır.
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch09_conveyor_pick_place import telemetry  # noqa: E402
from ch09_conveyor_pick_place.grasp import fit_surface  # noqa: E402


# --- telemetry._clean -------------------------------------------------------


def test_clean_numpy_scalars_and_arrays():
    out = telemetry._clean({
        "i": np.int32(7),
        "f": np.float64(1.5),
        "v": np.array([1.0, 2.0, 3.0]),
    })
    assert out == {"i": 7, "f": 1.5, "v": [1.0, 2.0, 3.0]}
    json.dumps(out)  # patlamamalı


def test_clean_replaces_non_finite_with_none():
    # NaN düzlem oturtmadan gelebiliyor (tümüyle geçersiz derinlik yaması).
    # json.dumps bunu `NaN` diye yazar; katı JSON okuyucular satırı reddeder.
    out = telemetry._clean({"a": float("nan"), "b": float("inf"), "c": 2.0})
    assert out == {"a": None, "b": None, "c": 2.0}
    assert "NaN" not in json.dumps(out)


def test_clean_keeps_nested_structure():
    out = telemetry._clean({"n": [{"x": np.float32(0.5)}, (1, 2)]})
    assert out == {"n": [{"x": pytest.approx(0.5)}, [1, 2]]}


def test_emit_is_silent_without_install():
    # Kayıt düğümü kapalıyken görev etkilenmemeli.
    assert telemetry.is_active() is False
    telemetry.emit("surface", rms_residual_mm=1.0)  # istisna fırlatmamalı


def test_emit_never_raises_on_bad_payload():
    class Kotu:
        def __repr__(self):
            raise RuntimeError("repr bile patlıyor")

    telemetry.emit("surface", x=Kotu())  # sessizce yutulmalı


# --- CsvSink ----------------------------------------------------------------


def test_csv_sink_flushes_every_row(tmp_path):
    from ch09_conveyor_pick_place.recorder_node import CsvSink

    path = tmp_path / "alt" / "x.csv"
    sink = CsvSink(path, ["a", "b"])
    sink.write({"a": 1, "b": 2})
    # Kapatmadan okunabilmeli: koşu kesilse bile satır diskte olmalı.
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows == [{"a": "1", "b": "2"}]
    sink.write({"a": 3, "b": 4, "fazla": 9})  # bilinmeyen alan düşmeli
    sink.close()
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [r["a"] for r in rows] == ["1", "3"]
    assert sink.rows == 2


def test_csv_sink_creates_nothing_until_first_row(tmp_path):
    from ch09_conveyor_pick_place.recorder_node import CsvSink

    path = tmp_path / "bos.csv"
    CsvSink(path, ["a"]).close()
    assert not path.exists()


# --- yama sayısı ------------------------------------------------------------


def test_fit_surface_reports_patch_and_band_counts():
    """
    Bandın kaç noktayı elediği ölçülebilir olmalı.

    inliers tek başına bandın işe yarayıp yaramadığını göstermez; oran için
    banda GİREN nokta sayısı da gerekir. Burada yamanın yarısı 1 m'de, yarısı
    1.5 m'de: bant yalnız çapanın bulunduğu yüzeyi tutmalı.
    """
    # Yüzey bir IZGARA olmalı, doğru değil: tek bir doğru üzerindeki noktalara
    # oturan düzlem tanımsızdır (en küçük iki tekil değer eşit) ve normalin
    # yönü rastgele çıkar. Gerçek yama da iki boyutludur.
    gx, gy = np.meshgrid(np.linspace(-0.02, 0.02, 8),
                         np.linspace(-0.02, 0.02, 8))
    duz = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 1.0)])
    arka = duz.copy()
    arka[:, 2] = 1.5
    points = np.vstack([duz, arka])

    patch = fit_surface(points, depth_band=0.05, min_points=25,
                        anchor=(0.0, 0.0, 1.0))
    assert patch is not None
    assert patch.patch_points == 128
    assert patch.inliers == 64  # yalnız çapa yüzeyi bantta kaldı
    assert patch.rms_residual < 1e-6
    assert math.isclose(abs(patch.normal[2]), 1.0, abs_tol=1e-6)
