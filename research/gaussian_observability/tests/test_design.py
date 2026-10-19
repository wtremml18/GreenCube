import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from design import condition, scores


def test_design_score_matches_actual_posterior_trace_reduction():
    rng = np.random.default_rng(4)
    f = rng.normal(size=(30, 16))
    c = f @ f.T + 0.1 * np.eye(30)
    target = np.arange(18, 30)
    values = scores(c, np.arange(6), target, 0.2)
    actual = []
    for i in range(6):
        m, after = condition(np.zeros(30), c, 3 * i + np.arange(3), np.zeros(3), 0.2)
        actual.append(
            np.trace(c[np.ix_(target, target)])
            - np.trace(after[np.ix_(target, target)])
        )
        assert np.linalg.eigvalsh(after).min() > 0
    np.testing.assert_allclose(values, actual, atol=1e-11)


def test_block_conditioning_matches_batch_and_order_is_irrelevant():
    rng = np.random.default_rng(19)
    f = rng.normal(size=(15, 15))
    c = f @ f.T + np.eye(15)
    y = rng.normal(size=6)
    m = np.zeros(15)
    mb, cb = condition(m, c, np.arange(6), y, 0.02)
    m1, c1 = condition(m, c, np.arange(3), y[:3], 0.02)
    ms, cs = condition(m1, c1, np.arange(3, 6), y[3:], 0.02)
    mr, cr = condition(m, c, np.arange(3, 6), y[3:], 0.02)
    mr, cr = condition(mr, cr, np.arange(3), y[:3], 0.02)
    np.testing.assert_allclose(ms, mb, atol=1e-11)
    np.testing.assert_allclose(cs, cb, atol=1e-11)
    np.testing.assert_allclose(mr, mb, atol=1e-11)
    np.testing.assert_allclose(cr, cb, atol=1e-11)


def test_scores_rotation_invariance():
    rng = np.random.default_rng(12)
    f = rng.normal(size=(30, 30))
    c = f @ f.T + np.eye(30)
    rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    r = np.kron(np.eye(10), rot)
    np.testing.assert_allclose(
        scores(c, np.arange(6), np.arange(18, 30), 0.1),
        scores(r @ c @ r.T, np.arange(6), np.arange(18, 30), 0.1),
        atol=1e-10,
    )


def test_conditioning_calibrated_on_prior_draws():
    rng = np.random.default_rng(89)
    f = rng.normal(size=(12, 12))
    c = f @ f.T + np.eye(12)
    truth = rng.multivariate_normal(np.zeros(12), c, size=30000)
    noise = rng.normal(0, 0.5, (len(truth), 3))
    obs = truth[:, :3] + noise
    gain = np.linalg.solve(c[:3, :3] + 0.25 * np.eye(3), c[:3]).T
    error = truth - obs @ gain.T
    _, post = condition(np.zeros(12), c, np.arange(3), np.zeros(3), 0.25)
    np.testing.assert_allclose(np.mean(error**2, axis=0), np.diag(post), rtol=0.035)
