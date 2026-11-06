import logging
import os
from typing import Any, Optional, Tuple

import numpy as np
import trimesh


class MeshAnalyzer:
    """
    Bakış noktası planlaması için şase STL dosyasını çözümler.

    Muayene edilecek yüzey noktalarını ve bu noktaların normallerini çıkarır.
    """

    def __init__(self, mesh_path: str, scale: float = 0.001,
                 logger: Optional[Any] = None) -> None:
        self.mesh_path = mesh_path
        self.scale = scale
        self.mesh = None
        # Bir rclpy düğüm günlükçüsü verilebilir; böylece kayıtlar sahibi olan
        # düğümün adı ve çıktısı altında görünür. Verilmezse bağımsız/komut
        # satırı/test kullanımı için standart `logging` modülü kullanılır.
        self.logger = logger or logging.getLogger(__name__)

    def load_mesh(self) -> None:
        if not os.path.exists(self.mesh_path):
            self.logger.error(f"Mesh file not found: {self.mesh_path}")
            raise FileNotFoundError(f"Mesh file not found: {self.mesh_path}")

        self.logger.info(f"Loading mesh from {self.mesh_path} (scale={self.scale})...")
        self.mesh = trimesh.load(self.mesh_path)

        if not hasattr(self.mesh, "faces") or len(self.mesh.faces) == 0:
            self.logger.error(
                "Loaded mesh has no faces. File may be empty/corrupt or not a surface mesh: "
                f"{self.mesh_path}"
            )
            raise ValueError(f"Mesh has no faces: {self.mesh_path}")

        # Ölçeği uygula (genellikle mm'den m'ye)
        self.mesh.apply_scale(self.scale)
        self.logger.info(
            f"Loaded mesh with {len(self.mesh.faces)} faces and {len(self.mesh.vertices)} "
            "vertices "
            f"(bounds after scaling: min={self.mesh.bounds[0].tolist()}, "
            f"max={self.mesh.bounds[1].tolist()}, "
            f"extents={self.mesh.extents.tolist()})."
        )

        if not self.mesh.is_watertight:
            self.logger.warning(
                "Mesh is not watertight. Surface sampling and normals may be unreliable "
                "at open edges; consider repairing chassis.stl."
            )

    def sample_surface(self, num_points: int = 15000, min_normal_z: float = -0.95
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Muayene hedef noktalarını elde etmek için ağ yüzeyinden örnek alır.

        Doğrudan aşağı bakan noktaları (örneğin alt yüzey) eler.
        """
        if self.mesh is None:
            self.load_mesh()

        self.logger.info(f"Sampling {num_points} points from the surface...")
        # Yüzeyden düzgün dağılımlı noktalar örneklemek için trimesh kullan
        points, face_indices = trimesh.sample.sample_surface(self.mesh, num_points)

        # Örneklenen noktaların yüz normallerini al
        normals = self.mesh.face_normals[face_indices]

        # min_normal_z'ye göre aşağı bakan noktaları ele
        # normal_z çok negatifse nokta aşağı bakıyordur (örneğin şasenin altı).
        valid_mask = normals[:, 2] >= min_normal_z

        points = points[valid_mask]
        normals = normals[valid_mask]
        face_indices = face_indices[valid_mask]

        dropped = num_points - len(points)
        self.logger.info(
            f"Retained {len(points)} points after filtering downward-facing surfaces "
            f"(dropped {dropped}, min_normal_z={min_normal_z})."
        )

        if len(points) == 0:
            self.logger.error(
                "All sampled points were filtered out. Check min_normal_z threshold and "
                "mesh orientation (normals may be inverted)."
            )
        elif len(points) < 0.1 * num_points:
            self.logger.warning(
                f"Only {len(points)}/{num_points} points retained after filtering. "
                "Coverage planning may under-sample the chassis; verify mesh normals/orientation."
            )

        return points, normals, face_indices

    def analyze_and_save(self, output_path: str, num_points: int = 15000,
                         min_normal_z: float = -0.95) -> Tuple[np.ndarray, np.ndarray]:
        points, normals, face_indices = self.sample_surface(num_points, min_normal_z)

        bounds = self.mesh.bounds
        mesh_bounds = {
            'min': bounds[0],
            'max': bounds[1],
            'center': self.mesh.centroid,
            'extents': self.mesh.extents
        }

        # Çıktı klasörü yoksa oluştur
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # npz olarak kaydet
        np.savez(
            output_path,
            points=points,
            normals=normals,
            face_indices=face_indices,
            mesh_bounds_min=mesh_bounds['min'],
            mesh_bounds_max=mesh_bounds['max'],
            mesh_center=mesh_bounds['center'],
            mesh_extents=mesh_bounds['extents']
        )
        self.logger.info(f"Saved analysis results to {output_path}")
        return points, normals


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Mesh Analyzer for Viewpoint Planning")
    parser.add_argument('--mesh', type=str, required=True, help="Path to chassis.stl")
    parser.add_argument('--output', type=str, default='chassis_analysis.npz', help="Output npz "
                                                                                   "file path")
    parser.add_argument('--samples', type=int, default=15000, help="Number of points to sample")
    parser.add_argument('--scale', type=float, default=0.001, help="Scale factor (default 0.001 "
                                                                   "for mm to m)")

    args = parser.parse_args()

    analyzer = MeshAnalyzer(args.mesh, scale=args.scale)
    analyzer.analyze_and_save(args.output, num_points=args.samples)
