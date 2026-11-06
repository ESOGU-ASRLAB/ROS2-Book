"""
HARMONY doğrulama senaryosunun yapay kusur (defect) tanımları.

Bu modül algılama (kusur yayını) ve temizleme (kusura gidiş) düğümlerince ortak
kullanılır; böylece koordinatlar, tipler, kimlikler ve gidilecek eklem açıları
tek yerde tanımlı kalır.

Kusurlar, kapı üzerinde fiziksel olarak bulunan ancak windowdoor.stl /
windowlessdoor.stl modellerinde yer almayan hataları temsil eder. Koordinatlar
`world` çerçevesinde ve metre cinsindendir.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from builtin_interfaces.msg import Time
    from visualization_msgs.msg import MarkerArray

# ---------------------------------------------------------------------------
# Kusur tanımları
# ---------------------------------------------------------------------------

#: (defect_id, defect_type, (x, y, z)) — world çerçevesi, metre
DEFECTS: List[Tuple[str, str, Tuple[float, float, float]]] = [
    ("DEF-01", "Protrusion", (-1.100, 0.800, 1.520)),
    ("DEF-02", "Protrusion", (-1.100, 0.300, 1.530)),
    ("DEF-03", "Dent",       (-0.918, 0.274, 1.040)),
    ("DEF-04", "Dent",       (-0.892, 1.680, 1.000)),
]

#: Kapıdan doğrusal eksene doğru (+X) ve yukarı (+Z) bırakılan pay [m].
STANDOFF_DX = 0.15
STANDOFF_DZ = 0.15

#: Sabit kamera yönelimi: optik +Z ekseni -X yönüne (kapıya) bakar.
#: ur10e_depth_optical_frame için world <- optical dönüşümünün kuaterniyonu.
CAMERA_QUAT_XYZW: Tuple[float, float, float, float] = (0.5, 0.5, -0.5, -0.5)

#: Ana başlangıç (home) konumu: algılama ve temizleme sonunda dönülen konfigürasyon.
#: SABİT — değiştirmeyin.
HOME_JOINTS: List[float] = [1.0, 0.0, -math.pi / 2, 0.0, -math.pi / 2, 0.0, 0.0]

#: Kusur başına gidilecek eklem açıları [ray(m), pan, lift, elbow, w1, w2, w3].
#:
#: tools/generate_defect_configs.py ile üretildi (dx=0.15, dz=0.15,
#: kamera yatayda -X'e bakar). Hedeflenen çerçeve ur10e_depth_optical_frame.
#:
#: DİKKAT: DEF-01 ve DEF-02 (z ~ 1.52 m) için +15 cm z payı robotun erişim
#: sınırının dışında kalıyor; aşağıdaki çözümler hedefe ~6 cm yaklaşabiliyor
#: (kamera z ~ 1.62 m'de kalıyor). Tam hedefe ulaşmak için x payını 0.30 m'ye
#: çıkarmak ya da bu iki kusur için z payını kaldırmak gerekir.
DEFECT_JOINTS: Dict[str, List[float]] = {
    # hedef kamera=(-0.950, 0.800, 1.670)  ulaşılan=(-0.916, 0.800, 1.619)
    # konum_hatası=61.5 mm, kapı mesafesi=22.0 cm
    "DEF-01": [0.645241, 2.907982, -1.009268, 0.044431, -0.662556, -1.551967, 2.914284],
    # hedef kamera=(-0.950, 0.300, 1.680)  ulaşılan=(-0.912, 0.300, 1.623)
    # konum_hatası=68.3 mm, kapı mesafesi=21.6 cm
    "DEF-02": [0.147541, 2.909280, -1.012262, 0.040188, -0.661241, -1.549926, 2.916590],
    # hedef kamera=(-0.768, 0.274, 1.190)  ulaşılan=(-0.767, 0.273, 1.190)
    # konum_hatası=1.1 mm, kapı mesafesi=18.6 cm
    "DEF-03": [0.472524, 0.705916, -2.004140, -1.118372, 1.551177, -1.569809, 0.707527],
    # hedef kamera=(-0.742, 1.680, 1.150)  ulaşılan=(-0.741, 1.679, 1.150)
    # konum_hatası=1.0 mm, kapı mesafesi=18.8 cm
    "DEF-04": [1.891474, 0.744085, -1.975678, -1.229000, 1.633109, -1.569839, 0.745605],
}

#: Temizleme sırası: doğrusal eksen hareketini azaltmak için y'ye göre artan.
CLEANING_ORDER: List[str] = ["DEF-03", "DEF-02", "DEF-01", "DEF-04"]

# Durum değerleri
STATUS_DETECTED = "DETECTED"
STATUS_CLEANING = "CLEANING"
STATUS_CLEANED = "CLEANED"
#: Hedefe gidilemedi (MoveIt planlayamadı / yürütme başarısız). Ayrı bir durum
#: olması önemli: DETECTED'a geri dönerse işaretçi kırmızı/turuncu kalır ve
#: "hiçbir şey olmamış" gibi görünür, hata gözden kaçar.
STATUS_FAILED = "FAILED"

# Konu adları
TOPIC_DEFECT = "/harmony/mock_perception/defect"           # birincil, kusur başına
TOPIC_DEFECT_LIST = "/harmony/mock_perception/defect_list"  # yedek, JSON dizisi
TOPIC_DEFECT_STATUS = "/harmony/defect_status"              # durum güncellemeleri
TOPIC_DEFECT_MARKERS = "/harmony/defect_markers"            # RViz MarkerArray


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def defect_ids() -> List[str]:
    return [d[0] for d in DEFECTS]


def defect_by_id(defect_id: str) -> Optional[Tuple[str, str, Tuple[float, float, float]]]:
    for d in DEFECTS:
        if d[0] == defect_id:
            return d
    return None


def approach_position(position: Sequence[float]) -> Tuple[float, float, float]:
    """Kusur noktasından robota doğru pay bırakılmış yaklaşma noktası."""
    return (float(position[0]) + STANDOFF_DX, float(position[1]), float(position[2]) + STANDOFF_DZ)


def defect_payload(
    defect_id: str,
    defect_type: str,
    position: Sequence[float],
    status: str = STATUS_DETECTED,
    frame_id: str = "world",
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Arayüzün ve temizleme düğümünün beklediği JSON gövdesi."""
    ax, ay, az = approach_position(position)
    return {
        "timestamp": timestamp or "",
        "defect_id": defect_id,
        "defect_type": defect_type,
        "frame_id": frame_id,
        "position": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])},
        "approach": {"x": ax, "y": ay, "z": az},
        "status": status,
    }


def all_payloads(status: str = STATUS_DETECTED, frame_id: str = "world",
                 timestamp: Optional[str] = None) -> List[Dict[str, Any]]:
    return [
        defect_payload(did, dtype, pos, status=status, frame_id=frame_id, timestamp=timestamp)
        for did, dtype, pos in DEFECTS
    ]


def status_payload(defect_id: str, status: str) -> str:
    return json.dumps({"defect_id": defect_id, "status": status})


# ---------------------------------------------------------------------------
# RViz görselleştirme
# ---------------------------------------------------------------------------
#: Etiket metni yüksekliği [m]. RViz'de karakter genişliği ~0.6 x yükseklik
#: olduğundan büyük değer metni yanlara taşırır (metin pozunda ORTALANIR).
LABEL_TEXT_SIZE = 0.05
#: Etiketin kusur küresinin ne kadar üstünde duracağı [m].
LABEL_HEIGHT_OFFSET = 0.09

# Tipe göre renkler (durum DETECTED iken); sonra durum renkleri baskın gelir.
_TYPE_COLORS = {
    "Protrusion": (0.90, 0.15, 0.15, 0.9),   # kırmızı
    "Dent":       (1.00, 0.55, 0.00, 0.9),   # turuncu
}
_STATUS_COLORS = {
    STATUS_CLEANING: (1.00, 0.90, 0.10, 0.95),  # sarı
    STATUS_CLEANED:  (0.15, 0.80, 0.30, 0.9),   # yeşil
    STATUS_FAILED:   (0.80, 0.20, 0.90, 0.95),  # mor — hedefe gidilemedi
}


def _color(defect_type: str, status: str) -> Tuple[float, float, float, float]:
    if status in _STATUS_COLORS:
        return _STATUS_COLORS[status]
    return _TYPE_COLORS.get(defect_type, (0.7, 0.7, 0.7, 0.9))


def build_empty_marker_array(frame_id: str = "world",
                             stamp: Optional[Time] = None) -> MarkerArray:
    """
    Bütün kusur işaretçilerini silen boş MarkerArray.

    Algılama tamamlanmadan önce yayımlanır. Böylece konu sürekli etkin kalır
    (RViz'de seçilebilir) ama kusurlar senaryodan önce görünmez.
    """
    from visualization_msgs.msg import Marker, MarkerArray

    array = MarkerArray()
    for ns in ("defect", "defect_approach", "defect_label"):
        m = Marker()
        m.header.frame_id = frame_id
        if stamp is not None:
            m.header.stamp = stamp
        m.ns = ns
        m.id = 0
        m.action = Marker.DELETEALL
        array.markers.append(m)
    return array


def build_marker_array(statuses: Optional[Dict[str, str]] = None,
                       frame_id: str = "world",
                       stamp: Optional[Time] = None) -> MarkerArray:
    """
    Kusurlar için MarkerArray üretir.

    Her kusur için üç işaretçi:
      * ``defect``          — kusur noktasında küre
      * ``defect_approach`` — yaklaşma noktasından kusura ok (gidilecek yönelim)
      * ``defect_label``    — "DEF-01 Protrusion (DETECTED)" metni

    statuses: defect_id -> durum. Verilmeyen kusurlar DETECTED sayılır.
    """
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker, MarkerArray

    statuses = statuses or {}
    array = MarkerArray()

    for idx, (did, dtype, pos) in enumerate(DEFECTS):
        status = statuses.get(did, STATUS_DETECTED)
        r, g, b, a = _color(dtype, status)
        ax, ay, az = approach_position(pos)

        sphere = Marker()
        sphere.header.frame_id = frame_id
        if stamp is not None:
            sphere.header.stamp = stamp
        sphere.ns = "defect"
        sphere.id = idx
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(pos[0])
        sphere.pose.position.y = float(pos[1])
        sphere.pose.position.z = float(pos[2])
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.06
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = r, g, b, a
        array.markers.append(sphere)

        # Yaklaşma oku: robotun duracağı noktadan kusura; okun yönü kameranın
        # bakış yönünü gösterir.
        arrow = Marker()
        arrow.header.frame_id = frame_id
        if stamp is not None:
            arrow.header.stamp = stamp
        arrow.ns = "defect_approach"
        arrow.id = idx
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose.orientation.w = 1.0
        start, end = Point(), Point()
        start.x, start.y, start.z = ax, ay, az
        end.x, end.y, end.z = float(pos[0]), float(pos[1]), float(pos[2])
        arrow.points = [start, end]
        arrow.scale.x = 0.012   # sap çapı
        arrow.scale.y = 0.028   # uç çapı
        arrow.scale.z = 0.04    # uç uzunluğu
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = r, g, b, 0.85
        array.markers.append(arrow)

        # TEXT_VIEW_FACING işaretçisi pozunda ORTALANIR; uzun tek satırlık metin
        # yanlara doğru metrelerce yayılıp başka kusurların/robotun üzerine
        # taşar. Bu yüzden metin kısa satırlara bölünür (en uzun satır ~9
        # karakter) ve doğrudan kürenin hemen üstüne yerleştirilir.
        label = Marker()
        label.header.frame_id = frame_id
        if stamp is not None:
            label.header.stamp = stamp
        label.ns = "defect_label"
        label.id = idx
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = float(pos[0])
        label.pose.position.y = float(pos[1])
        label.pose.position.z = float(pos[2]) + LABEL_HEIGHT_OFFSET
        label.pose.orientation.w = 1.0
        label.scale.z = LABEL_TEXT_SIZE
        label.color.r, label.color.g, label.color.b, label.color.a = 1.0, 1.0, 1.0, 0.95
        # Üç kısa satır: kimlik / tip / durum. Tek satıra birleştirilirse metin
        # yanlara metrelerce yayılır (bkz. LABEL_TEXT_SIZE açıklaması).
        label.text = f"{did}\n{dtype}\n{status}"
        array.markers.append(label)

    return array
