#!/usr/bin/env python3
"""
Eklem sınırlarını gözeten 2*pi eklem hedefi sarma.

NEDEN GEREKLİ
-------------
IK, bir çözüm ailesinin YALNIZCA BİR dalını döndürür. Dönel bir eklemde theta
ile theta +/- 2*pi AYNI fiziksel konfigürasyondur: bağlantı pozları, dolayısıyla
çarpışmalar ve kamera yönelimi aynıdır. Ama gidilecek SAYI olarak çok farklıdır.
Planlayıcı IK çözücünün döndürdüğü dalı olduğu gibi sakladığı için ardışık iki
bakış noktası örneğin +100 derece ve -260 derece olarak saklanabilir: aynı eklem
açısıdır, ama kol aralarında boş yere tam bir tur daha döner. Gerçek bir koşuda
bu, boşa giden muayene süresi, fazladan aşınma ve kablo sarılması demektir.

Çözüm, her hedef açıyı bir referans açıya (yürütme anında kolun ölçülen güncel
konumu ya da planlama anında bir önceki bakış noktası) EN YAKIN 2*pi eşdeğeriyle
yeniden ifade etmektir.

SINIR DENETİMİ NEDEN İŞİN ÖZÜDÜR
--------------------------------
Bu kuralın sınırları GÖZETMEYEN daha eski bir sürümü denendi ve geri alınmak
zorunda kalındı: sınır içindeki bir değeri (Kawasaki joint1 = 1.928 -> -4.355),
eklemin fiziksel olarak ulaşamayacağı bir değere çevirdi ve kol ilgisiz, yanlış
yönelimli pozlara gitti. Bu yüzden buradaki her aday YALNIZCA o eklemin URDF
sınırları (bir güvenlik payı düşülerek) içinde kalıyorsa kabul edilir. Böylece
dönüşüm, etkisiz kalması gereken yerde tam olarak etkisiz kalır:

    eklem aralığı < 2*pi   -> sınır içinde en fazla bir eşdeğer vardır -> değişiklik yok
                              (Kawasaki joint2/3/5 ve UR URDF'inin bilerek
                              +/-pi ile sınırladığı UR dirseği)
    eklem aralığı >= 2*pi  -> gerçek bir seçenek vardır -> en yakını alınır
                              (UR pan/lift/wrist1..3 ve Kawasaki joint4/joint6,
                              hepsi +/-2*pi)

Doğrusal eklemlere (UR rayı, world_to_agv) hiç dokunulmaz: metreler sarılmaz.

Bilerek saf Python + NumPy ile yazılmıştır (ROS içe aktarımı yoktur); böylece
planlayıcı düğümü, yürütücüler ve çevrim dışı araçlar tek bir uygulamayı paylaşır.
"""
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

TWO_PI = 2.0 * math.pi

# Konumu bir AÇI olan, dolayısıyla 2*pi ile periyodik eklem tipleri.
_REVOLUTE_TYPES = ("revolute", "continuous")


class JointLimit:
    """
    Bir eklemin sarma için gereken bilgileri: açı mı ve ne kadar gidebilir.

    `lower`/`upper` dönel eklemlerde radyandır. `continuous` eklemin URDF sınırı
    yoktur; (-inf, +inf) alır ve her eşdeğer kabul edilebilir.
    """

    __slots__ = ("name", "jtype", "lower", "upper")

    def __init__(self, name: str, jtype: str, lower: float, upper: float) -> None:
        self.name = name
        self.jtype = jtype
        self.lower = lower
        self.upper = upper

    @property
    def is_revolute(self) -> bool:
        return self.jtype in _REVOLUTE_TYPES

    @property
    def span(self) -> float:
        return self.upper - self.lower

    @property
    def wrappable(self) -> bool:
        """
        Sınır içinde ikinci bir eşdeğer olabiliyorsa True döndürür.

        Aralığı TAM OLARAK 2*pi olan bir eklem (Kawasaki joint1, +/-180 derece)
        pratikte sarılamaz: tek alternatif karşı sınırın kenarındadır ve güvenlik
        payı onu reddeder.
        """
        return self.is_revolute and self.span > TWO_PI

    def __repr__(self) -> str:
        return (f"JointLimit({self.name!r}, {self.jtype!r}, "
                f"{math.degrees(self.lower):.1f}deg, {math.degrees(self.upper):.1f}deg)")


def parse_joint_limits(urdf_xml: str) -> Dict[str, JointLimit]:
    """
    URDF metnini {eklem_adı: JointLimit} sözlüğüne ayrıştırır.

    Yalnızca gerçekten hareket eden eklemler döndürülür; sabit/mimic eklemler burada
    önemsizdir. <limit> etiketi olmayan dönel eklem tahmin edilmek yerine atlanır;
    böylece ayrıştırılamayan bir model kötü bir sarmaya değil, "sarma" kararına düşer.
    """
    limits = {}
    root = ET.fromstring(urdf_xml)
    for joint in root.iter("joint"):
        name = joint.get("name")
        jtype = (joint.get("type") or "").lower()
        if not name or jtype in ("fixed", "floating", "planar", ""):
            continue
        if jtype == "continuous":
            limits[name] = JointLimit(name, jtype, -math.inf, math.inf)
            continue
        tag = joint.find("limit")
        if tag is None:
            continue
        try:
            lower = float(tag.get("lower"))
            upper = float(tag.get("upper"))
        except (TypeError, ValueError):
            continue
        if upper < lower:
            lower, upper = upper, lower
        limits[name] = JointLimit(name, jtype, lower, upper)
    return limits


def wrap_angle_to(value: float, reference: float, limit: Optional[JointLimit],
                  margin: float = 0.05) -> float:
    """
    `value` değerinin `reference`'a en yakın, `limit` içindeki 2*pi eşdeğeri.

    Eklem doğrusalsa, kabul edilebilir eşdeğer yoksa ya da sınır bilgisi yoksa
    `value` değişmeden döner. [lower + margin, upper - margin] dışında bir açı
    asla döndürülmez.
    """
    if limit is None or not limit.is_revolute:
        return value
    lo = limit.lower + margin
    hi = limit.upper - margin
    if lo > hi:  # Pay, eklemin kendi aralığından geniş; ekleme dokunma
        return value

    # k = 0, `value` değerinin kendisidir. Gerçek her eklemi kapsayacak kadar dışa
    # doğru ara (+/-2 tur, buradaki en geniş +/-720 derece durumunu kapsar) ve
    # referansa en yakın kabul edilebilir adayı tut.
    best = None
    best_dist = math.inf
    for k in range(-3, 4):
        cand = value + k * TWO_PI
        if cand < lo or cand > hi:
            continue
        dist = abs(cand - reference)
        if dist < best_dist:
            best, best_dist = cand, dist
    if best is None:
        # `value` zaten sınır dışında (eskimiş ya da kötü tohumlanmış bir IK çözümü).
        # Dokunma: geçersiz bir hedefi sessizce taşımak sorunu gizlerdi.
        return value
    return best


# (ad, eski_değer, yeni_değer, kazanılan_radyan)
JointChange = Tuple[str, float, float, float]


def wrap_to_reference(goal: Sequence[float], names: Sequence[str],
                      reference: Mapping[str, float],
                      limits: Optional[Mapping[str, JointLimit]],
                      margin: float = 0.05, min_gain: float = 0.0
                      ) -> Tuple[List[float], List[JointChange]]:
    """
    Bir eklem hedefinin tamamını referans konfigürasyona doğru sarar.

    goal      -- hedef konumlar listesi (radyan / metre), `names` sırasıyla
    names     -- eklem adları, goal ile aynı uzunlukta
    reference -- {eklem_adı: ölçülen_ya_da_önceki_konum}; burada olmayan eklemlere
                 dokunulmaz (referans asla tahmin edilmez)
    limits    -- parse_joint_limits() çıktısı {eklem_adı: JointLimit}
    margin    -- her sınır kenarından uzak tutulan pay [radyan]
    min_gain  -- yalnızca eklemin yolunu en az bu kadar radyan kısaltan bir
                 değişiklik kabul edilir; böylece hedefler boşuna değiştirilmez

    (yeni_hedef, değişiklikler) döndürür; değişiklikler, en çok kazandırandan
    başlayarak (ad, eski_değer, yeni_değer, kazanılan_radyan) listesidir.
    """
    out = list(goal)
    changes = []
    for i, name in enumerate(names):
        limit = limits.get(name) if limits else None
        if limit is None or not limit.wrappable:
            continue
        if name not in reference:
            continue
        ref = float(reference[name])
        old = float(goal[i])
        new = wrap_angle_to(old, ref, limit, margin=margin)
        saved = abs(old - ref) - abs(new - ref)
        if abs(new - old) > 1e-9 and saved >= min_gain:
            out[i] = new
            changes.append((name, old, new, saved))
    changes.sort(key=lambda c: -c[3])
    return out, changes


def describe_changes(changes: Sequence[JointChange]) -> str:
    """wrap_to_reference() çıktısının tek satırlık, okunabilir özeti (derece)."""
    if not changes:
        return "no joints wrapped"
    parts = [f"{n} {math.degrees(o):.1f} -> {math.degrees(v):.1f} "
             f"(saves {math.degrees(s):.0f} deg)" for n, o, v, s in changes]
    total = sum(math.degrees(c[3]) for c in changes)
    return f"wrapped {len(changes)} joint(s), saving {total:.0f} deg total: " + "; ".join(parts)


def path_travel(goal: Sequence[float], names: Sequence[str],
                reference: Mapping[str, float], revolute_only: bool = True,
                limits: Optional[Mapping[str, JointLimit]] = None) -> float:
    """
    Bir referans konfigürasyon ile hedef arasındaki toplam |fark|.

    Sarmanın küçültmeye çalıştığı büyüklüktür. Günlük kaydında ve çevrim dışı
    önce/sonra karşılaştırmalarında kullanılır.
    """
    total = 0.0
    for i, name in enumerate(names):
        if name not in reference:
            continue
        if revolute_only:
            limit = limits.get(name) if limits else None
            if limit is not None and not limit.is_revolute:
                continue
        total += abs(float(goal[i]) - float(reference[name]))
    return total
