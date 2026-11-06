#!/usr/bin/env python3
"""
İki araba kapısını "parça" nokta bulutlarına örnekler; İNANÇ HARİTASI (belief map) girdisi.

Şase OctoMap hattı (pcd2octomap_builder) iki aşamada çalışır:

  1. inanç haritası : chassis_pcd/pcds/PART*.pcd; CAD parçası başına bir PCD, her biri
                      farklı renge boyanır ve BOŞ (free) olarak eklenir. Bu, "var olan ve
                      bu yüzden görülmesi gereken yüzey"in temel doğrusudur.
  2. doluluk        : toplanan algılayıcı bulutları o inanç haritasına damgalanır;
                      yalnızca bir inanç vokseline düşen noktalar sayılır ve parça başına
                      dolu/inanç voksel oranları kaplama raporunu oluşturur.

Kapıların da aynı 1. aşama girdisine ihtiyacı vardır; ama şasenin 57 PCD'si hazırdı,
çünkü CAD'i 57 ayrı parça STL'si olarak gelmişti. Kapılar iki STL'dir ve iki satırlık
bir kaplama raporu ("windowdoor %94") kaçırılan yerin NEREDE olduğunu söylemez. Bu
yüzden her kapı iki büyük ekseni (Y ve Z) boyunca bir yama ızgarasına bölünür ve her
yama bir "parça" olur. Rapor böylece kapının bir haritası gibi okunur ve octovis,
görülmemiş bir yamanın hemen göze çarptığı renkli bir yama örgüsü gösterir.

Kapı başına tek parça davranışı için patches_y = patches_z = 1 verin.

BİRİMLER: kapı STL'leri METRE cinsindendir (bkz. merge_door_meshes); milimetre olan
chassis.stl'den farklıdır. Noktalar bu yüzden metre cinsinden yazılır ve OctoMap
oluşturucu parts_scale=1.0 ile çalışmalıdır. Şase oluşturucu /1000 dönüşümünü sabit
kodlar; kapıların kendi oluşturucusunun olmasının sebebi tam olarak budur.

Çıktı bir derleme ürünüdür (~700 bin nokta); bu yüzden birleştirilmiş ağla aynı
önbellek klasöründe durur ve bir kaynak STL ya da yama düzeni değiştiğinde yeniden
üretilir.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from typing import List, Optional, Sequence, TYPE_CHECKING

from ch08_door_inspection.merge_door_meshes import DEFAULT_DOORS

if TYPE_CHECKING:
    import numpy as np  # yalnızca tip ipuçları için; çalışma anında tembel içe aktarılır

DEFAULT_PARTS_DIR = os.path.expanduser("~/.cache/ch08_door_inspection/door_parts")

# Metrekare başına nokta. Şase parçaları da aynı yoğunlukta örneklendi; 2 cm'lik
# OctoMap çözünürlüğünde yüzeyin değdiği her voksele ~40 örnek düşer; şanssız bir
# örnekleme yüzünden hiçbir inanç vokselinin kaçmayacağı kadar yoğundur.
DEFAULT_DENSITY = 100_000

# Kapı başına yama ızgarası. 1.2 x 1.5 m'lik bir kapıda 3 yatay (Y) x 4 dikey (Z) =
# ~40 x 38 cm'lik 12 yama; kaçırılan bir bakış noktasının gerçekten görüldüğü ölçek budur.
DEFAULT_PATCHES_Y = 3
DEFAULT_PATCHES_Z = 4

MANIFEST = "parts_manifest.json"


def _log(msg: str) -> None:
    print(f"[doors_parts_prep] {msg}", flush=True)


def _config_id(sources: Sequence[str], density: float, py: int, pz: int) -> str:
    """
    Çıktıyı değiştiren her şeyin parmak izi.

    Böylece yama düzenindeki bir değişiklik, değiştirilmiş bir STL gibi önbelleği geçersiz kılar.
    """
    h = hashlib.sha1()
    for s in sources:
        mtime = os.path.getmtime(s) if os.path.exists(s) else 0.0
        h.update(f"{s}:{mtime:.3f}".encode())
    h.update(f"{density}:{py}:{pz}".encode())
    return h.hexdigest()[:16]


def is_stale(parts_dir: str, config_id: str) -> bool:
    """parts_dir bu yapılandırmanın tamamlanmış derlemesini tutmuyorsa True döndürür."""
    try:
        with open(os.path.join(parts_dir, MANIFEST)) as f:
            return json.load(f).get("config_id") != config_id
    except (OSError, ValueError):
        return True


def write_pcd_binary(path: str, points: 'np.ndarray') -> None:
    """
    (N,3) boyutlu kayan noktalı bir diziyi ikili (binary) PCD olarak yazar.

    Şasenin kullandığı ASCII yerine ikili: aynı 700 bin nokta ~20 MB metin yerine 8 MB
    tutar ve PCL'in loadPCDFile'ı hangisi olduğu söylenmeden ikisini de okur.
    """
    import numpy as np

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {len(pts)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(pts)}\n"
        "DATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(pts.tobytes())


def build(
        sources: Optional[Sequence[str]] = None, parts_dir: Optional[str] = None,
        density: float = DEFAULT_DENSITY, patches_y: int = DEFAULT_PATCHES_Y,
        patches_z: int = DEFAULT_PATCHES_Z, force: bool = False) -> str:
    """Her kapıyı `parts_dir` altında yama PCD'lerine örnekler. O klasörü döndürür."""
    import numpy as np

    sources = list(sources or DEFAULT_DOORS)
    parts_dir = parts_dir or DEFAULT_PARTS_DIR
    patches_y = max(1, int(patches_y))
    patches_z = max(1, int(patches_z))

    missing = [s for s in sources if not os.path.exists(s)]
    if missing:
        raise FileNotFoundError(f"door mesh(es) not found: {missing}")

    config_id = _config_id(sources, density, patches_y, patches_z)
    if not force and not is_stale(parts_dir, config_id):
        _log(f"up to date, reusing {parts_dir}")
        return parts_dir

    import trimesh  # tembel içe aktarım: önbellek hazırken başlatma dosyası bu maliyeti ödemesin

    # Boştan yeniden üret. Eski yama dosyalarını bırakmak, önceki bir ızgaranın
    # parçalarını inanç haritasında sessizce tutardı ve kaplama raporu artık güncel
    # düzenin parçası olmayan yüzeyi sayardı.
    if os.path.isdir(parts_dir):
        shutil.rmtree(parts_dir)
    os.makedirs(parts_dir, exist_ok=True)

    total_pts = 0
    total_area = 0.0
    written = []

    for src in sources:
        door = os.path.splitext(os.path.basename(src))[0]
        mesh = trimesh.load(src, force="mesh")
        if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
            raise ValueError(f"mesh has no faces: {src}")

        n = max(10_000, int(mesh.area * density))
        pts, _ = trimesh.sample.sample_surface(mesh, n)
        pts = np.asarray(pts, dtype=np.float64)

        lo, hi = mesh.bounds
        # Bu kapının kendi sınır kutusu üzerinde eşit genişlikte bölmeler; böylece yama
        # ızgarası çifti değil kapıyı izler (iki kapı farklı Y'lerde durur).
        y_edges = np.linspace(lo[1], hi[1], patches_y + 1)
        z_edges = np.linspace(lo[2], hi[2], patches_z + 1)
        iy = np.clip(np.digitize(pts[:, 1], y_edges[1:-1]), 0, patches_y - 1)
        iz = np.clip(np.digitize(pts[:, 2], z_edges[1:-1]), 0, patches_z - 1)

        _log(f"{door}: faces={len(mesh.faces)} area={mesh.area:.3f} m^2 "
             f"-> {len(pts)} samples, {patches_y}x{patches_z} patches")
        total_area += float(mesh.area)

        for jy in range(patches_y):
            for jz in range(patches_z):
                sel = pts[(iy == jy) & (iz == jz)]
                if len(sel) == 0:
                    # Bir kapı dış hattı dikdörtgen değildir; köşe yamaları boş olabilir.
                    continue
                # "<kapı>__y<j>z<j>": oluşturucu raporunu "__" öncesindeki metne göre
                # gruplar; 24 yama satırını kapı başına 2 toplama çeviren budur.
                name = f"{door}__y{jy}z{jz}.pcd"
                write_pcd_binary(os.path.join(parts_dir, name), sel)
                written.append(name)
                total_pts += len(sel)

    with open(os.path.join(parts_dir, MANIFEST), "w") as f:
        json.dump({
            "config_id": config_id,
            "sources": sources,
            "density": density,
            "patches_y": patches_y,
            "patches_z": patches_z,
            "parts": sorted(written),
            "total_points": total_pts,
            "total_area_m2": round(total_area, 4),
            "units": "metres (build the octomap with parts_scale=1.0)",
        }, f, indent=2)

    _log(f"wrote {len(written)} part PCDs, {total_pts} points, {total_area:.3f} m^2 "
         f"-> {parts_dir}")
    return parts_dir


def ensure_parts(
        sources: Optional[Sequence[str]] = None, parts_dir: Optional[str] = None,
        density: float = DEFAULT_DENSITY, patches_y: int = DEFAULT_PATCHES_Y,
        patches_z: int = DEFAULT_PATCHES_Z, force: bool = False) -> str:
    """
    Asla hata fırlatmayan build().

    Böylece bir başlatma dosyası devam edebilir ve boş parça klasörünü oluşturucu kendi
    hata mesajıyla raporlar.
    """
    try:
        return build(sources, parts_dir, density, patches_y, patches_z, force)
    except Exception as e:                                   # noqa: BLE001
        _log(f"ERROR: {e}")
        return parts_dir or DEFAULT_PARTS_DIR


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--door", action="append", dest="doors", default=None,
                   help="source STL (repeatable); defaults to windowdoor + windowlessdoor")
    p.add_argument("--parts_dir", default=DEFAULT_PARTS_DIR,
                   help="where the patch PCDs are written")
    p.add_argument("--density", type=int, default=DEFAULT_DENSITY,
                   help="surface samples per square metre")
    p.add_argument("--patches_y", type=int, default=DEFAULT_PATCHES_Y,
                   help="patches across the door (1 = whole door is one part)")
    p.add_argument("--patches_z", type=int, default=DEFAULT_PATCHES_Z,
                   help="patches up the door (1 = whole door is one part)")
    p.add_argument("--force", action="store_true", help="rebuild even if up to date")
    a = p.parse_args(argv if argv is not None else sys.argv[1:])
    build(a.doors, a.parts_dir, a.density, a.patches_y, a.patches_z, a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
