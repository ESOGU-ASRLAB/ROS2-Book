import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from .mesh_analyzer import MeshAnalyzer

# Aday bakış noktası: {"position": np.ndarray(3), "rotation": np.ndarray(3, 3), ...}
Candidate = Dict[str, Any]


class ViewpointGenerator:
    """Hedef noktalardan aday bakış noktaları üretir ve kaplamayı hesaplar."""

    def __init__(self, mesh_analyzer: MeshAnalyzer,
                 config: Optional[Dict[str, Any]] = None,
                 logger: Optional[Any] = None) -> None:
        self.mesh_analyzer = mesh_analyzer
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or self.default_config()
        self.logger.info(f"ViewpointGenerator config: {self.config}")

    def default_config(self) -> Dict[str, Any]:
        """
        SICK Visionary-T Mini V3S145-1AAAAAA veri sayfasına dayanan varsayılan kamera modeli.

        Veri sayfası:
        https://cdn.sick.com/media/pdf/6/06/906/dataSheet_V3S145-1AAAAAA_1127086_en.pdf
          - Görüş alanı: 70 derece (yatay) x 60 derece (dikey). simrobot_ur_macro.xacro
            içinde depth_optical_frame üzerinde tanımlı Gazebo rgbd_camera algılayıcısıyla
            uyumludur (horizontal_fov=1.22 rad ~= 69.9 derece; 512x424 en-boy oranından
            türetilen dikey görüş alanı ~= 60 derece).
          - Algılayıcı çözünürlüğü: 512 x 424 (veri sayfasıyla birebir).
          - `fov_safety_margin_deg`, kaplama/örtülme denetimlerinde kullanılan *etkin*
            görüş alanını daraltır; böylece bakış noktaları, bozulmanın ve doğruluk
            kaybının arttığı mercek kenarından uzak tutulur. Görüş alanının tamamını
            kullanmak için 0.0 verin.
          - Gerçek donanım 30 fps'ye kadar çıkar; benzetimdeki algılayıcı işleme hattını
            gerçek zamanlı tutmak için 5 Hz ile sınırlanmıştır (bkz. simrobot_ur_macro.xacro
            içinde depth_optical_frame algılayıcısının update_rate değeri). Buradaki
            `frame_rate_hz` benzetim hızıyla aynıdır.
        """
        return {
            'horizontal_fov_deg': 70.0,
            'vertical_fov_deg': 60.0,
            'fov_safety_margin_deg': 3.0,
            'min_range_m': 0.2,
            'max_range_m': 1.0,
            'frame_rate_hz': 5.0,
            'viewpoint_distances': [0.3, 0.4, 0.5],
            'tilt_variations_deg': [-15, 0, 15],
            'max_incidence_angle_deg': 75.0
        }

    def generate_candidates(self, target_points: np.ndarray, target_normals: np.ndarray,
                            num_base_points: int = 250) -> List[Candidate]:
        """
        Hedef yüzey normallerine göre aday bakış noktaları üretir.

        num_base_points, kaç yüzey noktasının aday üreteceğini belirler. Bu, GEOMETRİK
        KAPLAMA TAVANININ ana ayarıdır: bir hedef ancak bir adayın görüş konisi ona
        ulaşıyorsa kaplanabilir; bu yüzden çok az taban noktası, büyük/karmaşık bir
        ağda boşluk bırakır. Artırmak tavanı yükseltir ama aday sayısını da artırır
        (daha yavaş kaplama matrisi ve daha çok IK sorgusu). Her taban noktası
        len(distances) * len(tilts) aday üretir.
        """
        candidates = []
        # Açıkça float64'e çevir: mesh_analyzer pratikte hep kayan noktalı diziler
        # üretir, ama tamsayı girdi (örneğin testlerden ya da elle kurulmuş
        # çağıranlardan) aşağıdaki yerinde bölme işlemini numpy'nin katı
        # same_kind dönüştürme kuralları yüzünden çökertirdi.
        target_points = np.asarray(target_points, dtype=np.float64)
        target_normals = np.asarray(target_normals, dtype=np.float64)

        # Hedef noktaları ~num_base_points aday tohumuna seyrelt.
        step = max(1, len(target_points) // max(1, num_base_points))

        base_points = target_points[::step]
        base_normals = target_normals[::step]

        for pt, normal in zip(base_points, base_normals):
            for d in self.config['viewpoint_distances']:
                # Kamera hedef noktaya BAKMALI; bu yüzden normal yönünde ilerlenir
                vp_pos = pt + normal * d

                # Varsayılan yönelim: kameranın Z ekseni -normal yönünü gösterir
                # ROS kamera çerçevelerinde: Z ileri, X sağa, Y aşağı
                z_axis = -normal

                # Yönelimi tam tanımlamak için bir yukarı vektör (Y ya da X ekseni) gerekir
                world_up = np.array([0, 0, 1])

                # Normalin world_up'a paralel olduğu durumu ele al
                if np.abs(np.dot(z_axis, world_up)) > 0.99:
                    world_up = np.array([1, 0, 0])

                x_axis = np.cross(world_up, z_axis)
                x_axis /= np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)

                base_rot_mat = np.column_stack((x_axis, y_axis, z_axis))

                for tilt in self.config['tilt_variations_deg']:
                    if tilt == 0:
                        candidates.append({'position': vp_pos, 'rotation': base_rot_mat})
                    else:
                        # Çeşitlemeler için X ya da Y ekseni etrafında eğ
                        r_tilt = Rotation.from_euler('x', tilt, degrees=True)
                        rot_mat = base_rot_mat @ r_tilt.as_matrix()
                        candidates.append({'position': vp_pos, 'rotation': rot_mat})

        self.logger.info(
            f"Generated {len(candidates)} candidate viewpoints from {len(base_points)} base "
            "surface points "
            f"(step={step}, distances={self.config['viewpoint_distances']}, "
            f"tilts={self.config['tilt_variations_deg']})."
        )
        return candidates

    def make_intersector(self) -> Any:
        """Örtülme denetimlerinde kullanılan trimesh ışın kesiştiricisini kurar."""
        if self.mesh_analyzer.mesh is None:
            self.mesh_analyzer.load_mesh()
        # Başarım için pyembree kurulu olmalı; değilse daha yavaş olan
        # yerleşik R-tree kesiştiricisine düşülür.
        try:
            intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(self.mesh_analyzer.mesh)
            self.logger.debug("Using pyembree-accelerated ray intersector.")
        except AttributeError:
            self.logger.warning(
                "pyembree not found. Falling back to the built-in R-tree ray intersector "
                "(this can be significantly slower for dense meshes/candidate sets)."
            )
            intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh_analyzer.mesh)
        return intersector

    def visible_mask(self, pos: np.ndarray, rot: np.ndarray, target_points: np.ndarray,
                     target_normals: np.ndarray, intersector: Any,
                     stats: Optional[Dict[str, int]] = None) -> np.ndarray:
        """
        Tek bir kamera pozunun hangi hedefleri görebildiğini gösteren mantıksal maske.

        Poz: pos; rot sütunları X/Y/Z, kamera +Z boyunca bakar. Menzil, görüş alanı,
        geliş açısı ve örtülme dikkate alınır. Hem tam kaplama matrisinde hem de
        kümeleyicinin birleştirilmiş pozları yeniden doğrulamasında kullanılır.
        `stats` (bir sözlük) verilirse ret sayıları sebebe göre biriktirilir.
        """
        pos = np.asarray(pos, dtype=np.float64)
        rot = np.asarray(rot, dtype=np.float64)

        margin = self.config.get('fov_safety_margin_deg', 0.0)
        hfov = np.deg2rad(self.config['horizontal_fov_deg'] - margin)
        vfov = np.deg2rad(self.config['vertical_fov_deg'] - margin)
        min_range = self.config['min_range_m']
        max_range = self.config['max_range_m']
        max_incidence = np.cos(np.deg2rad(self.config['max_incidence_angle_deg']))

        vecs = target_points - pos
        dists = np.linalg.norm(vecs, axis=1)
        dist_mask = (dists >= min_range) & (dists <= max_range)

        vecs_cam = vecs @ rot
        z_cam = vecs_cam[:, 2]
        z_cam_safe = np.where(z_cam > 1e-6, z_cam, 1e-6)
        x_angles = np.abs(np.arctan(vecs_cam[:, 0] / z_cam_safe))
        y_angles = np.abs(np.arctan(vecs_cam[:, 1] / z_cam_safe))
        fov_mask = (x_angles <= hfov / 2) & (y_angles <= vfov / 2) & (z_cam > 0)

        safe_dists = np.where(dists > 1e-9, dists, 1e-9)
        ray_dirs = vecs / safe_dists[:, np.newaxis]
        dot_prods = np.sum(ray_dirs * target_normals, axis=1)
        incidence_mask = dot_prods <= -max_incidence

        mask = dist_mask & fov_mask & incidence_mask

        if stats is not None:
            stats['distance'] = stats.get('distance', 0) + int(np.sum(~dist_mask))
            stats['fov'] = stats.get('fov', 0) + int(np.sum(dist_mask & ~fov_mask))
            stats['incidence'] = stats.get('incidence', 0) + int(
                np.sum(dist_mask & fov_mask & ~incidence_mask))

        # Örtülme denetimi: ışın boyunca ilk ağ kesişimi hedefin kendisinden
        # daha yakınsa hedef gizlidir (kendine çarpma için 2 cm tolerans).
        visible_indices = np.where(mask)[0]
        pre_occlusion_count = len(visible_indices)
        if pre_occlusion_count > 0:
            ray_origins = np.tile(pos, (pre_occlusion_count, 1))
            ray_directions = ray_dirs[visible_indices]
            locations, index_ray, index_tri = intersector.intersects_location(
                ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False
            )
            for r_idx, hit_loc in zip(index_ray, locations):
                orig_idx = visible_indices[r_idx]
                if np.linalg.norm(hit_loc - pos) < dists[orig_idx] - 0.02:
                    mask[orig_idx] = False

        if stats is not None:
            stats['occluded'] = stats.get('occluded', 0) + (
                pre_occlusion_count - int(np.sum(mask)))
            stats['visible'] = stats.get('visible', 0) + int(np.sum(mask))
        return mask

    def compute_coverage_matrix(self, candidates: List[Candidate], target_points: np.ndarray,
                                target_normals: np.ndarray) -> np.ndarray:
        """
        Hangi hedefin hangi adaydan görünür olduğunu gösteren mantıksal matrisi hesaplar.

        Matrisin boyutu [num_candidates, num_targets]'tır.
        """
        self.logger.info(
            f"Computing coverage matrix for {len(candidates)} candidates x {len(target_points)} "
            "targets. "
            "This may take a while..."
        )
        num_c = len(candidates)
        num_t = len(target_points)
        coverage_matrix = np.zeros((num_c, num_t), dtype=bool)
        target_points = np.asarray(target_points, dtype=np.float64)
        target_normals = np.asarray(target_normals, dtype=np.float64)

        intersector = self.make_intersector()

        # Bütün adaylar boyunca ret sayılarını sebebe göre biriktirir.
        stats = {}
        for i, cand in enumerate(candidates):
            coverage_matrix[i, :] = self.visible_mask(
                cand['position'], cand['rotation'], target_points, target_normals,
                intersector, stats=stats)
            if (i + 1) % 50 == 0 or (i + 1) == num_c:
                self.logger.info(f"Processed {i + 1}/{num_c} candidates...")

        occluded_total = stats.get('occluded', 0)
        visible_total = stats.get('visible', 0)
        rejected_by_distance = stats.get('distance', 0)
        rejected_by_fov = stats.get('fov', 0)
        rejected_by_incidence = stats.get('incidence', 0)

        candidates_with_no_coverage = int(np.sum(coverage_matrix.sum(axis=1) == 0))
        self.logger.info(
            f"Coverage matrix complete. Total visible (point,candidate) pairs: {visible_total}, "
            f"occluded/rejected by ray-trace: {occluded_total}. "
            f"{candidates_with_no_coverage}/{num_c} candidates cover zero target points "
            "(will never be selected by the optimizer)."
        )

        # BELİRLEYİCİ tanı: robot yok sayıldığında ne kadar kaplama *mümkün*? En az
        # bir adayın gördüğü hedef geometrik olarak kaplanabilir; geri kalanı, IK ve
        # erişilebilirlik ne kadar iyi olursa olsun asla kaplanamaz. Bu, bir *algılama
        # geometrisi* darboğazını (tavanı daha çok aday / daha geniş menzil / daha iyi
        # bakış açıları / daha temiz bir ağ ile yükseltin) bir *erişilebilirlik*
        # darboğazından (IK/montaj/doğrusal eksen düzeltilir) ayırır.
        targets_covered_by_any = int(np.sum(coverage_matrix.sum(axis=0) > 0))
        geometric_ceiling = targets_covered_by_any / num_t if num_t else 0.0
        self.logger.info(
            f"GEOMETRIC COVERAGE CEILING: {targets_covered_by_any}/{num_t} "
            f"({geometric_ceiling * 100:.1f}%) targets are visible from at least one candidate, "
            "ignoring reachability. No plan can exceed this. If it is far below your "
            "coverage_threshold, the bottleneck is SENSING GEOMETRY (candidate density / range / "
            "FOV / occlusion / mesh), not the robot. If it is high but achieved coverage is low, "
            "the bottleneck is REACHABILITY (IK / mounting / linear axis)."
        )
        total_pairs = num_c * num_t
        self.logger.info(
            f"Rejection breakdown over {total_pairs} (candidate,target) pairs -- "
            f"out of range: {rejected_by_distance}, outside FOV: {rejected_by_fov}, "
            f"incidence angle too steep (>{self.config['max_incidence_angle_deg']} deg): "
            f"{rejected_by_incidence}, "
            f"occluded: {occluded_total}, visible: {visible_total}. "
            "Use this to decide which camera parameter to relax (range/FOV/max_incidence) if "
            "coverage is low."
        )
        if candidates_with_no_coverage == num_c:
            self.logger.error(
                "No candidate viewpoint covers any target point. Check camera FOV/range config, "
                "mesh scale, and viewpoint_distances -- coverage optimization will fail."
            )

        return coverage_matrix
