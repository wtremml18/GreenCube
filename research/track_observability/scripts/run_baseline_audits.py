"""New analytic benchmarks, not reproduction of unavailable MATLAB code."""

import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.kernels import fit, tune, folds


def dipole(x):
    r = np.linalg.norm(x, axis=1)
    m = np.array([0.0, 0.0, 1.0])
    return 3 * x * x[:, 2, None] / r[:, None] ** 5 - m / r[:, None] ** 3


def numerical_div(fun, x, dims, h=1e-5):
    ans = np.zeros(len(x))
    for j in dims:
        step = np.zeros_like(x)
        step[:, j] = h
        ans += (fun(x + step)[:, j] - fun(x - step)[:, j]) / (2 * h)
    return ans


rng = np.random.default_rng(1806)
x = rng.uniform([0.5, -0.5, 0.5], [1.5, 0.5, 1.5], (96, 3))
q = rng.uniform([0.55, -0.45, 0.55], [1.45, 0.45, 1.45], (600, 3))
y = dipole(x)
truth = dipole(q)
models = {}
for kind in ["df", "independent"]:
    m, b, _ = tune(x, y, folds(len(x), seed=5), kind, lengths=(0.2, 0.4, 0.8, 1.6))
    models["3d_" + kind] = dict(
        mse=float(np.mean(np.sum((m.predict(q) - truth) ** 2, axis=1))), **b
    )
# Meridional slice y=0, using components (Bx,Bz), away from source.
x[:, 1] = 0
q[:, 1] = 0
y = dipole(x)
truth = dipole(q)
for kind in ["df", "independent"]:
    m, b, _ = tune(
        x[:, [0, 2]],
        y[:, [0, 2]],
        folds(len(x), seed=5),
        kind,
        lengths=(0.2, 0.4, 0.8, 1.6),
    )
    models["2d_slice_" + kind] = dict(
        mse=float(
            np.mean(np.sum((m.predict(q[:, [0, 2]]) - truth[:, [0, 2]]) ** 2, axis=1))
        ),
        **b,
    )
analytic = -3 * q[:, 2] / np.linalg.norm(q, axis=1) ** 5
models["derivative_check"] = dict(
    full3d_max_abs=float(np.max(np.abs(numerical_div(dipole, q, [0, 1, 2])))),
    slice_formula_max_error=float(
        np.max(np.abs(numerical_div(dipole, q, [0, 2]) - analytic))
    ),
    slice_div_rms=float(np.sqrt(np.mean(analytic**2))),
)
# Input permutation and isotropic change of units at fixed hyperparameters.
z = rng.uniform(size=(40, 3))
v = rng.normal(size=(40, 3))
qq = rng.uniform(size=(50, 3))
perm = rng.permutation(len(z))
a = fit(z, v, 0.3, 0.01)
b = fit(z[perm], v[perm], 0.3, 0.01)
c = fit(1000 * z, v, 300.0, 0.01)
models["invariance"] = dict(
    permutation_max_abs=float(np.max(np.abs(a.predict(qq) - b.predict(qq)))),
    units_max_abs=float(np.max(np.abs(a.predict(qq) - c.predict(1000 * qq)))),
)
(ROOT / "results/baseline_audits.json").write_text(json.dumps(models, indent=2) + "\n")
print(json.dumps(models, indent=2))
