import pathlib
import pytest
import numpy as np
from ch07_viewpoint_planner.mesh_analyzer import MeshAnalyzer


def test_mesh_analyzer_initialization() -> None:
    analyzer = MeshAnalyzer("dummy_path.stl", scale=0.001)
    assert analyzer.mesh_path == "dummy_path.stl"
    assert analyzer.scale == 0.001
    assert analyzer.mesh is None


def test_load_nonexistent_mesh() -> None:
    analyzer = MeshAnalyzer("nonexistent.stl")
    with pytest.raises(FileNotFoundError):
        analyzer.load_mesh()


def test_sample_surface(tmp_path: pathlib.Path) -> None:
    import trimesh
    # Basit bir kutu ağı oluştur
    mesh = trimesh.creation.box(extents=(1000, 1000, 1000))  # mm cinsinden 1 m'lik kutu
    mesh_path = tmp_path / "test_box.stl"
    mesh.export(str(mesh_path))

    analyzer = MeshAnalyzer(str(mesh_path), scale=0.001)
    # 1000 nokta
    points, normals, face_indices = analyzer.sample_surface(num_points=1000, min_normal_z=-0.9)

    assert points is not None
    assert normals is not None
    assert len(points) == len(normals)
    assert len(points) > 0
    # Kutunun aşağı bakan yüzleri var (z normali = -1); bunlar elenmeli
    assert np.all(normals[:, 2] >= -0.9)
