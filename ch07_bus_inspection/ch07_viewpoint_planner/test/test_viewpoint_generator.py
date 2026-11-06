import numpy as np
from ch07_viewpoint_planner.viewpoint_generator import ViewpointGenerator


class MockMeshAnalyzer:
    def __init__(self) -> None:
        self.mesh = None


def test_viewpoint_generator_initialization() -> None:
    vg = ViewpointGenerator(MockMeshAnalyzer())
    # Varsayılan görüş alanı SICK Visionary-T Mini V3S145-1AAAAAA veri sayfasıyla
    # (70 x 60 derece) ve simrobot_ur_macro.xacro'daki Gazebo rgbd_camera ile uyumludur.
    assert vg.config['horizontal_fov_deg'] == 70.0
    assert vg.config['vertical_fov_deg'] == 60.0


def test_generate_candidates() -> None:
    vg = ViewpointGenerator(MockMeshAnalyzer())
    target_points = np.array([[0, 0, 0]])
    target_normals = np.array([[0, 0, 1]])

    candidates = vg.generate_candidates(target_points, target_normals)
    # uzaklıklar: 0.3, 0.4, 0.5 (3 uzaklık)
    # eğimler: -15, 0, 15 (3 eğim)
    # Toplam aday = 1 * 3 * 3 = 9
    assert len(candidates) == 9

    # Aday konumunun normal doğrultusunda olup olmadığını denetle
    pos0 = candidates[0]['position']
    assert np.allclose(pos0, np.array([0, 0, 0.3]))
