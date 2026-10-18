import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.fields import tracks, target_grid, noise_audit

q = target_grid()
rows = []
for cutoff in [3, 4, 5]:
    designs = []
    for n in [12, 24, 48, 96]:
        x, _ = tracks(n)
        designs.append(("parallel", len(x), 0.0, x))
    x, _ = tracks(24)
    for jitter in [0.0001, 0.001, 0.01]:
        z = x.copy()
        z[:, 1] += np.random.default_rng(81).normal(0, jitter, len(z))
        designs.append(("jittered", len(z), jitter, z))
    for extra in [4, 8, 16, 32]:
        z = np.vstack(
            [
                x,
                np.random.default_rng(82).uniform(
                    [0.025, 0.125], [0.975, 0.875], (extra, 2)
                ),
            ]
        )
        designs.append(("off_track", len(z), 0.0, z))
    for design, n, jitter, z in designs:
        for delta in [0.001, 0.01, 0.05]:
            a = noise_audit(z, q, cutoff, delta)
            rows.append(
                dict(
                    design=design,
                    n=n,
                    jitter=jitter,
                    cutoff=cutoff,
                    delta=delta,
                    **{k: v for k, v in a.items() if k != "coefficients"},
                )
            )
pd.DataFrame(rows).to_csv(ROOT / "results/noise_observability.csv", index=False)
