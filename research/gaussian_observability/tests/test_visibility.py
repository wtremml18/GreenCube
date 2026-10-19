import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visibility import project_points, world_to_voxel


def test_optical_axis_and_opengl_vertical_sign():
    uv, depth = project_points(
        np.array([[0.0, 0, -2], [1.0, 1, -2]]), np.eye(4), 100, 80, np.pi / 2
    )
    np.testing.assert_allclose(uv, [[49.5, 39.5], [74.5, 14.5]])
    np.testing.assert_allclose(depth, [2, 2])


def test_camera_translation_changes_projection_consistently():
    rng = np.random.default_rng(9)
    world = rng.normal(size=(10, 3))
    world[:, 2] -= 3
    c = np.eye(4)
    shift = np.array([0.2, -0.1, 0.5])
    c[:3, 3] = shift
    uv1, z1 = project_points(world, np.eye(4), 320, 180, 0.4)
    uv2, z2 = project_points(world + shift, c, 320, 180, 0.4)
    np.testing.assert_allclose(uv1, uv2)
    np.testing.assert_allclose(z1, z2)


def test_voxel_world_roundtrip_respects_axis_order_and_anisotropic_extent():
    rng = np.random.default_rng(83)
    xyz = rng.uniform(0, 1, (20, 3)) * [100, 150, 100]
    rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = [0.2, -0.1, 0.7]
    scale = np.array([0.49, 0.735, 0.49])
    meta = dict(voxel_matrix=m.tolist(), voxel_scale=scale.tolist())
    actual = m[:, [2, 1, 0, 3]]
    world = (xyz / [100, 150, 100] * scale) @ actual[:3, :3].T + actual[:3, 3]
    np.testing.assert_allclose(world_to_voxel(world, meta), xyz, atol=1e-12)
