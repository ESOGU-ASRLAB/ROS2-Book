#!/usr/bin/env python3
"""
API anahtarının nereden okunduğunu sabitleyen testler.

Anahtar yalnızca ortam değişkeninden okunduğu sürece, düğüm .bashrc okumayan
her bağlamda (masaüstü kısayolu, IDE terminali, systemd, .bashrc düzenlenmeden
önce açılmış terminal) sessizce çalışmıyordu ve belirti "her açılışta anahtarı
yeniden girmek gerekiyor" gibi görünüyordu. Dosya yolu bu yüzden var; testler
sıralamayı ve ayrıştırmayı koruyor.

    python3 -m pytest test/test_api_key.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ch09_conveyor_pick_place import er_client  # noqa: E402


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_env_wins_over_file(monkeypatch, tmp_path):
    key_file = tmp_path / "api_key"
    key_file.write_text("dosyadaki-anahtar\n")
    monkeypatch.setattr(er_client, "API_KEY_PATHS", (str(key_file),))
    monkeypatch.setenv("GEMINI_API_KEY", "ortamdaki-anahtar")
    assert er_client.load_api_key() == "ortamdaki-anahtar"


def test_falls_back_to_file(clean_env, monkeypatch, tmp_path):
    key_file = tmp_path / "api_key"
    key_file.write_text("dosyadaki-anahtar\n")
    monkeypatch.setattr(er_client, "API_KEY_PATHS", (str(key_file),))
    assert er_client.load_api_key() == "dosyadaki-anahtar"


def test_google_api_key_also_accepted(clean_env, monkeypatch):
    monkeypatch.setattr(er_client, "API_KEY_PATHS", ())
    monkeypatch.setenv("GOOGLE_API_KEY", "ikinci-degisken")
    assert er_client.load_api_key() == "ikinci-degisken"


@pytest.mark.parametrize("content, expected", [
    ("duz-anahtar\n", "duz-anahtar"),
    ("  bosluklu  \n", "bosluklu"),
    ('GEMINI_API_KEY="tirnakli"\n', "tirnakli"),
    ("export GEMINI_API_KEY=export-bicimi\n", "export-bicimi"),
    ("# yorum satiri\n\nyorumdan-sonraki\n", "yorumdan-sonraki"),
])
def test_file_formats(clean_env, monkeypatch, tmp_path, content, expected):
    """Dosya hem düz anahtar hem source edilebilir biçimde yazılabilmeli."""
    key_file = tmp_path / "api_key"
    key_file.write_text(content)
    monkeypatch.setattr(er_client, "API_KEY_PATHS", (str(key_file),))
    assert er_client.load_api_key() == expected


def test_missing_everywhere_returns_none(clean_env, monkeypatch, tmp_path):
    monkeypatch.setattr(er_client, "API_KEY_PATHS", (str(tmp_path / "yok"),))
    assert er_client.load_api_key() is None


def test_empty_env_var_is_not_a_key(clean_env, monkeypatch, tmp_path):
    """
    Boş bir export, dosyayı gölgelememeli.

    'export GEMINI_API_KEY=' yazan bir satır ortamı boş bir değerle DOLDURUR;
    varlığına bakan bir kontrol bunu geçerli sayar ve hata anlaşılmaz olur.
    """
    key_file = tmp_path / "api_key"
    key_file.write_text("dosyadaki-anahtar\n")
    monkeypatch.setattr(er_client, "API_KEY_PATHS", (str(key_file),))
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    assert er_client.load_api_key() == "dosyadaki-anahtar"
