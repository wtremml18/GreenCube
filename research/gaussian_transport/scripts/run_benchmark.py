import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "track_observability"))
import numpy as np
import pandas as pd
from transport import advect, flow
from src.kernels import tune, folds

rows = []
for seed in range(20):
    rng = np.random.default_rng(20000 + seed)
    x = rng.uniform(-1, 1, (48, 3))
    centers = rng.uniform(-0.45, 0.45, (12, 3))
    cov = np.broadcast_to(
        np.diag([0.02**2, 0.03**2, 0.04**2]), (len(centers), 3, 3)
    ).copy()
    noise = rng.normal(0, 0.01, x.shape)
    for compressible in [False, True]:
        exact = lambda p: flow(p, compressible)
        mu, sigma, f = advect(exact, centers, cov, steps=64)
        for kind in ["df", "independent", "helmholtz"]:
            model, best, _ = tune(
                x,
                exact(x) + noise,
                folds(len(x), seed=seed),
                "df" if kind == "df" else kind,
                lengths=(0.3, 0.6, 1.2, 2.4),
            )
            pred, pcov, pf = advect(model.predict, centers, cov, steps=32)
            logdet = np.linalg.slogdet(pf)[1]
            rows.append(
                dict(
                    seed=seed,
                    flow="expanding" if compressible else "solenoidal",
                    kind=kind,
                    center_rmse=float(
                        np.sqrt(np.mean(np.sum((pred - mu) ** 2, axis=1)))
                    ),
                    covariance_relative_error=float(
                        np.mean(
                            np.linalg.norm(pcov - sigma, axis=(1, 2))
                            / np.linalg.norm(sigma, axis=(1, 2))
                        )
                    ),
                    mean_abs_logdet=float(np.mean(np.abs(logdet))),
                    logdet_error=float(
                        np.mean(np.abs(logdet - np.linalg.slogdet(f)[1]))
                    ),
                    **best,
                )
            )
    if seed % 5 == 0:
        print("transport seed", seed, flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results/transport.csv", index=False)
pd.DataFrame(rows).groupby(["flow", "kind"]).mean(numeric_only=True).to_csv(
    ROOT / "results/summary.csv"
)
