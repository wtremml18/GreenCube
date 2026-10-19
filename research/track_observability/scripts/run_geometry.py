import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.fields import tracks, target_grid, random_field
from src.geometry import push, geometry, tune_warped
from src.kernels import folds

rows = []
r, _ = tracks(24)
r = r[:, ::-1]
q = target_grid()
x = push(r, 0.12)
target = push(q, 0.12)
t = geometry(x, 0.12, 0.8)[1]
tq = geometry(target, 0.12, 0.8)[1]
for seed in range(20):
    clean = np.einsum("nij,nj->ni", t, random_field(r, seed))
    truth = np.einsum("nij,nj->ni", tq, random_field(q, seed))
    y = clean + np.random.default_rng(90000 + seed).normal(0, 0.05, clean.shape)
    for name, a, beta in [
        ("correct", 0.12, 0.8),
        ("underestimated", 0.08, 0.8),
        ("straight", 0.0, 0.8),
        ("reversed", -0.12, 0.8),
        ("naive_velocity", 0.0, 0.0),
    ]:
        m, b = tune_warped(x, y, folds(len(x), seed=seed), a, beta)
        rows.append(
            dict(
                seed=seed,
                geometry=name,
                mse=np.mean(np.sum((m.predict(target) - truth) ** 2, axis=1)),
                **b,
            )
        )
    print("geometry", seed, flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results/geometry.csv", index=False)
