import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from selection import block_ols, wendland_blocks


def test_block_ols_matches_bruteforce_refits():
    rng = np.random.default_rng(8)
    a, y = rng.normal(size=(31, 7, 3)), rng.normal(size=(31, 2))
    selected, trace = block_ols(a, y, 5)
    chosen = []
    for j, row in zip(selected, trace):
        losses = {}
        for c in set(range(7)) - set(chosen):
            x = a[:, chosen + [c]].reshape(31, -1)
            losses[c] = np.sum((y - x @ np.linalg.lstsq(x, y, rcond=None)[0]) ** 2)
        assert j == min(losses, key=losses.get)
        assert np.isclose(row["residual_sse"], losses[j], atol=1e-10)
        chosen.append(j)


def test_block_selection_invariant_to_coefficient_rotation_and_scale():
    rng = np.random.default_rng(9)
    a, y = rng.normal(size=(60, 9, 3)), rng.normal(size=(60, 3))
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    transformed = (
        np.einsum("nci,ij->ncj", a, rotation) * np.geomspace(0.1, 10, 9)[None, :, None]
    )
    assert block_ols(a, y, 5)[0] == block_ols(transformed, y, 5)[0]


def test_redundant_and_zero_blocks_are_handled():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(30, 4, 3))
    a[:, 2] = a[:, 0]
    a[:, 3] = 0
    y = a[:, 0] @ np.array([1, 2, 3])
    selected, trace = block_ols(a, y, 3)
    assert selected[0] in [0, 2]
    assert trace[0]["residual_sse"] < 1e-24
    assert len(set(selected)) == 3


def test_wendland_constraint_support_and_rotation():
    rng = np.random.default_rng(3)
    x, c, h, w = (
        rng.normal(size=(30, 3)),
        rng.normal(size=(4, 3)),
        np.array([2.0, 3.0, 4.0, 5.0]),
        rng.normal(size=(4, 3)),
    )

    def f(q):
        return np.einsum("ncij,cj->ni", wendland_blocks(q, c, h), w)

    divergence = np.zeros(len(x))
    eps = 1e-5
    for j in range(3):
        d = np.eye(3)[j] * eps
        divergence += (f(x + d)[:, j] - f(x - d)[:, j]) / (2 * eps)
    assert np.max(np.abs(divergence)) < 1e-7
    assert np.all(wendland_blocks(np.ones((1, 3)) * 100, c, h) == 0)
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    transformed = np.einsum(
        "ncij,cj->ni", wendland_blocks(x @ rotation, c @ rotation, h), w @ rotation
    )
    assert np.allclose(transformed, f(x) @ rotation)


def test_regularized_block_selection_matches_ridge_objective():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(25, 6, 3))
    y = rng.normal(size=(25, 2))
    ridge = 2.0
    augmented = np.concatenate((a, np.sqrt(ridge) * np.eye(18).reshape(18, 6, 3)))
    target = np.concatenate((y, np.zeros((18, 2))))
    selected, trace = block_ols(augmented, target, 4)
    chosen = []
    for actual, row in zip(selected, trace):
        losses = {}
        for c in set(range(6)) - set(chosen):
            x = a[:, chosen + [c]].reshape(25, -1)
            beta = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
            losses[c] = np.sum((y - x @ beta) ** 2) + ridge * np.sum(beta**2)
        assert actual == min(losses, key=losses.get)
        assert np.isclose(row["residual_sse"], losses[actual], atol=1e-10)
        chosen.append(actual)
