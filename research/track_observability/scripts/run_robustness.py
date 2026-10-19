"""Exploratory sensitivity checks planned after primary results; no test tuning."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.kernels import tune, folds
from run_public import quadrants


def dipole(x):
    r = np.linalg.norm(x, axis=1)
    return (
        3 * x * x[:, 2, None] / r[:, None] ** 5
        - np.array([0.0, 0.0, 1.0]) / r[:, None] ** 3
    )


def main():
    rows = []
    for seed in range(10):
        rng = np.random.default_rng(50000 + seed)
        x = rng.uniform([0.5, -0.5, 0.5], [1.5, 0.5, 1.5], (96, 3))
        q = rng.uniform([0.55, -0.45, 0.55], [1.45, 0.45, 1.45], (600, 3))
        noise = rng.normal(size=x.shape)
        for dim in [3, 2]:
            xx = x.copy()
            qq = q.copy()
            if dim == 2:
                xx[:, 1] = 0
                qq[:, 1] = 0
            ix = [0, 1, 2] if dim == 3 else [0, 2]
            clean = dipole(xx)[:, ix]
            truth = dipole(qq)[:, ix]
            rms = np.sqrt(np.mean(clean**2))
            for sigma in [0.0, 0.05]:
                y = clean + sigma * rms * noise[:, ix]
                for kind in ["df", "independent", "helmholtz"]:
                    m, b, _ = tune(
                        xx[:, ix],
                        y,
                        folds(len(x), seed=seed),
                        kind,
                        lengths=(0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
                        ridges=(1e-6, 1e-4, 0.01, 0.1),
                    )
                    rows.append(
                        dict(
                            seed=seed,
                            dim=dim,
                            noise_fraction=sigma,
                            kind=kind,
                            mse=np.mean(
                                np.sum((m.predict(qq[:, ix]) - truth) ** 2, axis=1)
                            ),
                            **b,
                        )
                    )
        print("dipole", seed, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/dipole_robustness.csv", index=False)
    rows = []
    predictions = []
    for dataset in ["gladfull", "laserfull"]:
        xy = np.loadtxt(ROOT / "data" / f"XY_train_{dataset}.csv", delimiter=",")
        uv = np.loadtxt(ROOT / "data" / f"UV_train_{dataset}.csv", delimiter=",")
        for seed in range(5):
            selected = np.sort(
                np.random.default_rng(20261018 + seed).choice(
                    len(xy), min(160, len(xy)), replace=False
                )
            )
            x = xy[selected]
            y = uv[selected]
            for fold, (tr, te) in enumerate(folds(len(x), quadrants(x))):
                origin = x[tr].mean(axis=0)
                scale = np.ptp(x[tr], axis=0).max()
                xx = (x - origin) / scale
                inner = folds(len(tr), quadrants(xx[tr]))
                baseline = np.mean(np.sum((y[te] - y[tr].mean(axis=0)) ** 2, axis=1))
                for kind in ["df", "independent", "helmholtz"]:
                    m, b, _ = tune(
                        xx[tr],
                        y[tr],
                        inner,
                        kind,
                        lengths=(0.04, 0.08, 0.16, 0.32, 0.64, 1.28),
                        ridges=(1e-4, 0.01, 0.1, 1.0),
                    )
                    pred = m.predict(xx[te])
                    mse = np.mean(np.sum((pred - y[te]) ** 2, axis=1))
                    rows.append(
                        dict(
                            dataset=dataset,
                            seed=seed,
                            fold=fold,
                            kind=kind,
                            n_test=len(te),
                            mse=mse,
                            mean_baseline_mse=baseline,
                            **b,
                        )
                    )
                    predictions.extend(
                        dict(
                            dataset=dataset,
                            seed=seed,
                            fold=fold,
                            kind=kind,
                            row=int(selected[i]),
                            u_pred=float(p[0]),
                            v_pred=float(p[1]),
                        )
                        for i, p in zip(te, pred)
                    )
            print("public sensitivity", dataset, seed, flush=True)
            pd.DataFrame(rows).to_csv(
                ROOT / "results/public_robustness.csv", index=False
            )
            pd.DataFrame(predictions).to_csv(
                ROOT / "results/public_robustness_predictions.csv", index=False
            )


if __name__ == "__main__":
    main()
