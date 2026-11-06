#!/usr/bin/env python3
"""
Araba kapısı STL'lerini kaplama planlayıcısı için TEK bir hedef ağda birleştirir.

Neden yeni bir planlayıcı yerine birleştirilmiş bir dosya: `multirobot_planner_node`
zaten ağdan bağımsızdır (`mesh_path` / `mesh_scale` parametredir) ama tam olarak BİR
ağı çözümler. Kapı işi iki kapı üzerinde TEK bir ortak plan istediği için oraya
varmanın en ucuz dürüst yolu ona ikisini birden içeren tek bir ağ vermektir.
ch07_viewpoint_planner ya da ch07_multirobot_planner içinde hiçbir şey değişmez; bu da
şase hattının donanımda doğrulanmış olanla bayt bayt aynı kaldığı anlamına gelir.

Kapı başına planlamak yerine birleştirmek örtülmeyi de doğru yapar: tek bir ağda her
kapı diğerine giden ışınları gerçek çift gibi engeller.

BİRİMLER: kapı STL'leri zaten METRE cinsindendir (URDF onları scale="1 1 1" ile
yükler; milimetre olan chassis.stl'den farklı). Değiştirilmeden yazılırlar; bu yüzden
planlayıcı mesh_scale=1.0 ile çalışmalıdır. Burada 0.001 vermek kapıları 2 mm'lik bir
noktaya küçültür ve her bakış noktası erişilemez olurdu.

Birleştirilmiş dosya kaynak değil, derleme ürünüdür: ~45 MB'tır (iki kapı birlikte
~900 bin üçgen taşır); bu yüzden depo dışındaki bir önbellek klasörüne yazılır ve bir
kaynak STL daha yeni olduğunda kendiliğinden yeniden üretilir.
"""
import argparse
import os
from typing import List, Optional, Sequence
import sys

from ch08_door_inspection import paths

# Kapı ağları araştırma deposundaki ur_description paketindedir
DOOR_MESH_DIR = paths.door_mesh_dir()

DEFAULT_DOORS = [
    os.path.join(DOOR_MESH_DIR, "windowdoor.stl"),
    os.path.join(DOOR_MESH_DIR, "windowlessdoor.stl"),
]
DEFAULT_OUTPUT = os.path.expanduser("~/.cache/ch08_door_inspection/doors_merged.stl")


def _log(msg: str) -> None:
    print(f"[doors_mesh_prep] {msg}", flush=True)


def is_stale(output: str, sources: Sequence[str]) -> bool:
    """`output` yeniden üretilmeliyse True döndürür: yoksa ya da herhangi bir kaynaktan eskiyse."""
    if not os.path.exists(output):
        return True
    out_mtime = os.path.getmtime(output)
    return any(os.path.getmtime(s) > out_mtime for s in sources if os.path.exists(s))


def merge(
        sources: Optional[Sequence[str]] = None, output: Optional[str] = None,
        force: bool = False) -> str:
    """
    Kapı ağlarını `output` içinde birleştirir. Çıktı yolunu döndürür.

    Hiçbir dönüşüm uygulanmaz: her STL kendi dünya yerleşimini zaten taşır (kapıları
    tutan URDF eklemleri birim dönüşümdür); bu yüzden yalnızca birleştirmek, iki kapıyı
    birbirlerine ve iki kola göre gerçekte durdukları yere koyar.
    """
    sources = list(sources or DEFAULT_DOORS)
    output = output or DEFAULT_OUTPUT

    missing = [s for s in sources if not os.path.exists(s)]
    if missing:
        raise FileNotFoundError(f"door mesh(es) not found: {missing}")

    if not force and not is_stale(output, sources):
        _log(f"up to date, reusing {output}")
        return output

    import trimesh  # tembel içe aktarım: başlatma dosyaları her çalıştırmada bu maliyeti ödemesin

    parts = []
    for s in sources:
        m = trimesh.load(s, force="mesh")
        if not hasattr(m, "faces") or len(m.faces) == 0:
            raise ValueError(f"mesh has no faces: {s}")
        lo, hi = m.bounds
        _log(f"{os.path.basename(s):22s} faces={len(m.faces):7d} area={m.area:6.3f} m^2 "
             f"bounds x[{lo[0]:+.3f},{hi[0]:+.3f}] y[{lo[1]:+.3f},{hi[1]:+.3f}] "
             f"z[{lo[2]:+.3f},{hi[2]:+.3f}]")
        parts.append(m)

    merged = trimesh.util.concatenate(parts)
    lo, hi = merged.bounds
    _log(f"merged: faces={len(merged.faces)} area={merged.area:.3f} m^2 "
         f"bounds x[{lo[0]:+.3f},{hi[0]:+.3f}] y[{lo[1]:+.3f},{hi[1]:+.3f}] "
         f"z[{lo[2]:+.3f},{hi[2]:+.3f}] (METRES; run the planner with mesh_scale=1.0)")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    tmp = output + ".tmp"
    merged.export(tmp, file_type="stl")
    os.replace(tmp, output)
    _log(f"wrote {output} ({os.path.getsize(output) / 1e6:.1f} MB)")
    return output


def ensure_merged(
        sources: Optional[Sequence[str]] = None, output: Optional[str] = None,
        force: bool = False) -> str:
    """
    Asla hata fırlatmayan merge(); yolu yine de döndürür.

    Böylece başlatma devam edebilir ve eksik ağı planlayıcı kendi hata mesajıyla raporlar.
    """
    try:
        return merge(sources, output, force)
    except Exception as e:                                   # noqa: BLE001
        _log(f"ERROR: {e}")
        return output or DEFAULT_OUTPUT


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--door", action="append", dest="doors", default=None,
                   help="source STL (repeatable); defaults to windowdoor + windowlessdoor")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="merged STL path")
    p.add_argument("--force", action="store_true", help="rebuild even if up to date")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    merge(args.doors, args.output, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
