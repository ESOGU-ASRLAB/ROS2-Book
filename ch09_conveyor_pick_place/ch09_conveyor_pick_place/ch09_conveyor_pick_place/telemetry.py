#!/usr/bin/env python3
"""
Ölçüm veri yolu: düğümlerin İÇİNDEKİ sayıları kayıt düğümüne taşır.

NEDEN AYRI BİR TOPIC: makaleye giren sayıların çoğu zaten hesaplanıyor ama
yalnızca `get_logger().info()` ile yazılıyordu; yani ancak terminal kaydından,
elle, gözle okunabiliyorlardı. Bir koşunun ardından "yama kaç noktaydı, kaçı
bantta kaldı, artık kaç mm'ydi" sorusunun cevabı ekran görüntüsünde kalıyor.
Burada aynı sayılar `/gemini/record` üzerine JSON olarak basılır ve
recorder_node onları CSV'ye açar.

`/gemini/status` YETMEZ: o, göreve bakan bir durum akışıdır (insan için kısa
metin). Ölçüm akışı ayrı tutulur ki durum mesajlarının biçimi bozulmadan
istenildiği kadar alan eklenebilsin.

TASARIM KURALLARI
  - Kurulmadıysa `emit` SESSİZCE hiçbir şey yapmaz. Kayıt düğümü kapalıyken
    ya da testlerde düğüm nesnesi yokken görev etkilenmemelidir.
  - `emit` HİÇBİR KOŞULDA istisna fırlatmaz. Ölçüm toplamak, görevi
    düşürmeye değecek bir iş değildir.
  - Yayıncı süreç başına tektir; locator, pick_place ve kavrayıcı aynı
    süreçte yaşadığı için ayrı ayrı yayıncı açmanın anlamı yok.
"""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # yalnızca tip ipuçları için; çalışma anında içe aktarılmaz
    from rclpy.node import Node

TOPIC = "/gemini/record"

_lock = threading.Lock()
_publisher = None  # rclpy Publisher
_node = None  # sahibi düğüm (yalnız logger için)
_run_id: Optional[str] = None


def install(node: Node, topic: str = TOPIC) -> None:
    """Veri yolunu bu düğüm üzerinden aç. İkinci çağrı yok sayılır."""
    global _publisher, _node
    with _lock:
        if _publisher is not None:
            return
        try:
            from std_msgs.msg import String
            _publisher = node.create_publisher(String, topic, 50)
            _node = node
        except Exception:  # noqa: BLE001 - ölçüm görevi düşürmez
            _publisher = None
            return
    node.get_logger().info(f"telemetri açık -> {topic}")


def set_run_id(run_id: str) -> None:
    """Koşu kimliği; her olayla birlikte gider, dosya adlarını eşleştirir."""
    global _run_id
    _run_id = str(run_id) if run_id else None


def is_active() -> bool:
    return _publisher is not None


def emit(kind: str, **fields: Any) -> None:
    """Tek bir ölçüm olayı yayınlar. Asla istisna fırlatmaz."""
    pub = _publisher
    if pub is None:
        return
    try:
        from std_msgs.msg import String
        payload = {
            "kind": str(kind),
            "wall": time.time(),
            "run": _run_id,
        }
        payload.update(_clean(fields))
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False, default=str)
        pub.publish(msg)
    except Exception:  # noqa: BLE001 - ölçüm görevi düşürmez
        pass


def _clean(value: Any) -> Any:
    """
    Değeri JSON'a çevrilebilir hâle getirir.

    numpy skalerleri ve dizileri, NaN/Inf ve demetler burada düzleşir; aksi
    hâlde `json.dumps` ya patlar ya da JSON okuyucuların reddettiği `NaN`
    üretir (recorder tarafında satır satır ayrıştırma bozulur).
    """
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # numpy'yi burada import etmemek için ördek tipi
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _clean(tolist())
        except Exception:  # noqa: BLE001
            return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _clean(item())
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)
