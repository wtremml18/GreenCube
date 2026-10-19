"""Secondary, explicitly post-primary geometry intervention experiment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.fields import tracks, target_grid, random_field
from src.kernels import tune, folds

q = target_grid()
rows = []
for seed in range(40):
    x, _ = tracks(24)
    truth = random_field(q, seed)
    y = random_field(x, seed) + np.random.default_rng(1000 + seed).normal(
        0, 0.05, x.shape
    )
    for extra in [8, 32]:
        for design in ["along", "off"]:
            rng = np.random.default_rng(9000 + seed)
            add = rng.uniform([0.025, 0.125], [0.975, 0.875], size=(extra, 2))
            if design == "along":
                add[:, 1] = np.tile([0.125, 0.375, 0.625, 0.875], extra // 4)
            noise = np.random.default_rng(10000 + seed).normal(0, 0.05, add.shape)
            z = np.vstack([x, add])
            v = np.vstack([y, random_field(add, seed) + noise])
            model, best, _ = tune(z, v, folds(len(z), seed=4000 + seed), "df")
            mse = np.mean(np.sum((model.predict(q) - truth) ** 2, axis=1))
            rows.append(dict(seed=seed, extra=extra, design=design, mse=mse, **best))
pd.DataFrame(rows).to_csv(ROOT / "results/design.csv", index=False)
