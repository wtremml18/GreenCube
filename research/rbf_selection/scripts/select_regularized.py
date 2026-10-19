"""Exploratory regularized OLS, with ridge strength selected from sensor CV.

Uses the same linearized image dictionary and locations; never reads GT volumes.
The coefficient prior is a surrogate, not the upstream physical regularizer.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from selection import block_ols, wendland_blocks

start = time.perf_counter()
z = np.load(ROOT / "results/autodiff_dictionary.npz")
centers, radii, jac = z["centers"], z["radii"], z["jacobian"]
target = np.load(ROOT / "results/selection_observations.npz")["image_target"]
sensors = json.loads(
    (
        ROOT.parent / "gaussian_observability/results/image_sensors_visual16.json"
    ).read_text()
)
x = np.array(sensors["0"]["xyz"])
y = np.stack([np.array(sensors[str(f)]["velocity"]).ravel() for f in range(3)], axis=1)
a = wendland_blocks(x, centers, radii).transpose(0, 2, 1, 3).reshape(48, -1)
labels = np.empty(16, dtype=int)
labels[np.random.default_rng(713).permutation(16)] = np.arange(16) % 4
records = []
for alpha in np.logspace(-6, 2, 9):
    for fold in range(4):
        tr = np.flatnonzero(np.repeat(labels != fold, 3))
        va = np.flatnonzero(np.repeat(labels == fold, 3))
        gram = a[tr] @ a[tr].T
        lam = alpha * np.trace(gram) / len(gram)
        coef = a[tr].T @ np.linalg.solve(gram + lam * np.eye(len(gram)), y[tr])
        records.append(
            {
                "relative_ridge": alpha,
                "fold": fold,
                "component_mse": float(np.mean((a[va] @ coef - y[va]) ** 2)),
            }
        )
cv = pd.DataFrame(records)
cv.to_csv(ROOT / "results/regularized_cv.csv", index=False)
alpha = float(cv.groupby("relative_ridge").component_mse.mean().idxmin())
# Normalize the entire sensor loss by 48, matching the joint selection objective.
ridge = alpha * np.trace(a @ a.T) / 48**2
image = jac / np.sqrt(len(jac))
sensor = a.reshape(48, len(centers), 3) / np.sqrt(48)
features = np.concatenate(
    [
        image,
        sensor,
        np.sqrt(ridge) * np.eye(3 * len(centers)).reshape(-1, len(centers), 3),
    ]
)
values = np.concatenate(
    [target / np.sqrt(len(jac)), y / np.sqrt(48), np.zeros((3 * len(centers), 3))]
)
order, trace = block_ols(features, values, 128)
for budget in [32, 64, 128]:
    ii = np.array(order[:budget])
    (ROOT / "results" / f"basis_joint_rols_{budget}.json").write_text(
        json.dumps(
            {
                "method": "joint_rols",
                "budget": budget,
                "centers": centers[ii].tolist(),
                "radii": radii[ii].tolist(),
                "candidate_indices": ii.tolist(),
            },
            indent=2,
        )
        + "\n"
    )
(ROOT / "results/joint_rols_trace.json").write_text(
    json.dumps(
        {
            "relative_ridge": alpha,
            "coefficient_ridge": ridge,
            "selection_seconds": time.perf_counter() - start,
            "steps": trace,
        },
        indent=2,
    )
    + "\n"
)
print(
    "Selected ridge",
    alpha,
    "coefficient penalty",
    ridge,
    "elapsed",
    time.perf_counter() - start,
)
