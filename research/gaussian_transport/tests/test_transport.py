import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm
from transport import advect


def test_affine_transport_matches_matrix_exponential():
    a = np.array([[0.2, 0.4, 0.0], [-0.1, -0.3, 0.2], [0.0, 0.0, 0.1]])
    x = np.array([[0.1, 0.2, 0.3], [-0.2, 0.3, 0.1]])
    cov = np.broadcast_to(np.diag([0.01, 0.02, 0.03]), (2, 3, 3))
    mu, sigma, f = advect(lambda z: z @ a.T, x, cov, duration=1, steps=32)
    e = expm(a)
    assert_allclose(mu, x @ e.T, atol=1e-9)
    assert_allclose(f, np.broadcast_to(e, f.shape), atol=1e-9)
    assert_allclose(sigma, e @ cov @ e.T, atol=1e-9)
    assert_allclose(np.linalg.det(f), 1.0, atol=1e-9)


def test_expansion_is_not_incompressible():
    x = np.ones((2, 3)) * 0.1
    cov = np.broadcast_to(np.eye(3) * 0.01, (2, 3, 3))
    _, sigma, f = advect(lambda z: 0.3 * z, x, cov, duration=0.5)
    assert_allclose(np.linalg.slogdet(f)[1], 0.45, atol=1e-9)
    assert_allclose(np.linalg.det(sigma) / np.linalg.det(cov), np.exp(0.9), rtol=1e-8)


def test_density_preserving_nonrigid_motion_and_sensor_rank():
    from ambiguity import generator, flow_map, sensor_operator

    scales = np.array([0.6, 0.35, 0.2])
    sigma = np.diag(scales**2)
    a = generator(scales, [0.3, 0.5, 1.0])
    f = flow_map(scales, [0.3, 0.5, 1.0], 2.0)
    assert abs(np.trace(a)) < 1e-14
    assert np.linalg.norm(a + a.T) > 0.1  # not just a rigid rotation
    assert np.allclose(f @ sigma @ f.T, sigma, atol=1e-14)
    assert np.linalg.matrix_rank(sensor_operator(np.zeros((1, 3)), scales)) == 0
    assert (
        np.linalg.matrix_rank(sensor_operator(np.array([[0.2, 0.1, 0.05]]), scales))
        == 2
    )
    assert (
        np.linalg.matrix_rank(
            sensor_operator(np.array([[0.2, 0.1, 0.05], [-0.1, 0.15, 0.1]]), scales)
        )
        == 3
    )
