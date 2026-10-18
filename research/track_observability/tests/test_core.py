import numpy as np
from numpy.testing import assert_allclose
from src.kernels import kernel, fit, divergence, folds
from src.fields import tracks, blind_field, null_audit, target_grid, basis


def test_kernel_psd_symmetry_normalization():
    rng = np.random.default_rng(12)
    for d in [2, 3]:
        x = rng.uniform(size=(16, d))
        for kind in ["df", "independent", "helmholtz"]:
            k = kernel(x, x, 0.25, kind, 0.3)
            assert_allclose(k, k.T, atol=1e-14)
            assert np.linalg.eigvalsh(k).min() > -1e-11
            assert_allclose(np.diag(k), 1.0)


def test_divergence_convergence_and_nonzero_control():
    rng = np.random.default_rng(4)
    for d in [2, 3]:
        x = rng.uniform(size=(24, d))
        y = rng.normal(size=(24, d))
        q = rng.uniform(size=(17, d))
        df = fit(x, y, 0.3, 0.01)
        e1 = np.max(np.abs(divergence(df, q, h=1e-3)))
        e2 = np.max(np.abs(divergence(df, q, h=1e-4)))
        assert e2 < e1 / 50
        assert e2 < 1e-4
        assert np.max(np.abs(divergence(fit(x, y, 0.3, 0.01, "independent"), q))) > 0.1


def test_blind_field_and_null_budget():
    x, _ = tracks(24)
    q = target_grid()
    assert np.max(np.abs(blind_field(x))) < 1e-12
    assert np.sqrt(np.mean(np.sum(blind_field(q) ** 2, axis=1))) > 1
    a = null_audit(x, q, 4)
    assert a["nullity"] > 0 and a["radius"] > 0.01
    assert a["residual"] < 1e-8
    assert_allclose(np.linalg.norm(a["coefficients"]), 1.0)
    assert_allclose(
        np.linalg.norm(basis(q) @ a["coefficients"]) / np.sqrt(len(q)), a["radius"]
    )


def test_group_splits_and_training_only_mean():
    x, g = tracks(5)
    for tr, te in folds(len(x), g):
        assert not set(g[tr]) & set(g[te])
    y = np.tile([4.0, -2.0], (len(x), 1))
    assert_allclose(
        fit(x, y, 0.2, 0.01).predict(target_grid()), np.tile([4.0, -2.0], (775, 1))
    )


def test_fourier_basis_has_zero_divergence():
    rng = np.random.default_rng(9)
    q = rng.uniform(size=(12, 2))
    c = rng.normal(size=basis(q).shape[1])
    h = 1e-5
    div = np.zeros(len(q))
    for j in range(2):
        step = np.zeros_like(q)
        step[:, j] = h
        div += (
            (basis(q + step) @ c).reshape(-1, 2)[:, j]
            - (basis(q - step) @ c).reshape(-1, 2)[:, j]
        ) / (2 * h)
    assert np.max(np.abs(div)) < 1e-6


def test_noise_budget_and_radius_monotonicity():
    from src.fields import noise_audit

    x, _ = tracks(24)
    q = target_grid()
    a = noise_audit(x, q, delta=0.01)
    assert a["coefficient_norm"] <= 1 + 1e-8 and a["observation_rms"] <= 0.01 + 1e-8
    assert_allclose(
        a["coefficient_norm"] ** 2 + (a["observation_rms"] / 0.01) ** 2, 1, atol=1e-8
    )
    assert_allclose(
        np.linalg.norm(basis(q) @ a["coefficients"]) / np.sqrt(len(q)), a["radius"]
    )
    assert a["radius"] >= null_audit(x, q)["radius"] - 1e-7
    assert noise_audit(x, q, delta=0.05)["radius"] >= a["radius"]


def test_piola_weighted_divergence_and_background():
    from src.geometry import geometry, push, warped_kernel, WarpedModel

    rng = np.random.default_rng(33)
    x = push(rng.uniform(size=(18, 2)), 0.12)
    k = warped_kernel(x, x, 0.3, 0.12, 0.8)
    assert np.allclose(k, k.T, atol=1e-13)
    assert np.linalg.eigvalsh(k).min() > -1e-12
    q = push(rng.uniform(size=(25, 2)), 0.12)
    m = WarpedModel(x, rng.normal(size=(18, 2)), 0.3, 0.01, 0.12, 0.8)

    def flux(p):
        return np.exp(0.8 * p[:, 0, None]) * m.predict(p)

    div = np.zeros(len(q))
    ordinary = np.zeros(len(q))
    h = 1e-5
    for j in range(2):
        d = np.zeros_like(q)
        d[:, j] = h
        div += (flux(q + d)[:, j] - flux(q - d)[:, j]) / (2 * h)
        ordinary += (m.predict(q + d)[:, j] - m.predict(q - d)[:, j]) / (2 * h)
    assert np.max(np.abs(div)) < 2e-5
    assert np.sqrt(np.mean(ordinary**2)) > 0.01
    b = np.array([0.7, -0.2])
    y = geometry(x, 0.12, 0.8)[1] @ b
    m = WarpedModel(x, y, 0.3, 0.01, 0.12, 0.8)
    assert np.allclose(m.predict(q), geometry(q, 0.12, 0.8)[1] @ b, atol=1e-11)


def test_prediction_equivariance_permutation_units_and_rotation():
    rng = np.random.default_rng(501)
    x = rng.normal(size=(25, 3))
    y = rng.normal(size=(25, 3))
    q = rng.normal(size=(12, 3))
    perm = rng.permutation(len(x))
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    for kind in ["df", "independent", "helmholtz"]:
        p = fit(x, y, 0.7, 0.03, kind, 0.4).predict(q)
        assert_allclose(
            fit(x[perm], y[perm], 0.7, 0.03, kind, 0.4).predict(q), p, atol=1e-11
        )
        assert_allclose(
            fit(x * 1000, y, 700.0, 0.03, kind, 0.4).predict(q * 1000), p, atol=1e-11
        )
        assert_allclose(
            fit(x @ rotation, y @ rotation, 0.7, 0.03, kind, 0.4).predict(q @ rotation),
            p @ rotation,
            atol=1e-11,
        )
