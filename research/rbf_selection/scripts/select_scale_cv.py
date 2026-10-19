"""Sensor-location CV selects one initial support scale; no withheld truth.

The surrogate fits linear RBF coefficients with ridge regularization. Final
camera training still initializes weights to zero and learns all parameters.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from selection import wendland_blocks

sensors = json.loads(
    (
        ROOT.parent / "gaussian_observability/results/image_sensors_visual16.json"
    ).read_text()
)
x = np.array(sensors["0"]["xyz"])
y = np.stack([np.array(sensors[str(f)]["velocity"]).ravel() for f in range(3)], axis=1)
labels = np.empty(16, dtype=int)
labels[np.random.default_rng(713).permutation(16)] = np.arange(16) % 4
records = []
for budget in [32, 64, 128]:
    start = time.perf_counter()
    base = json.loads((ROOT / "results" / f"basis_poisson_{budget}.json").read_text())
    centers = np.array(base["centers"])
    radius = np.array(base["radii"])
    scores = []
    for multiplier in [0.5, 1.0, 2.0]:
        matrix = (
            wendland_blocks(x, centers, radius * multiplier)
            .transpose(0, 2, 1, 3)
            .reshape(48, -1)
        )
        for ridge in [1e-4, 0.01, 1.0]:
            errors = []
            for fold in range(4):
                tr = np.flatnonzero(np.repeat(labels != fold, 3))
                va = np.flatnonzero(np.repeat(labels == fold, 3))
                a, b = matrix[tr], matrix[va]
                gram = a @ a.T
                penalty = ridge * np.trace(gram) / len(gram)
                coeff = a.T @ np.linalg.solve(gram + penalty * np.eye(len(gram)), y[tr])
                loss = float(np.mean((b @ coeff - y[va]) ** 2))
                records.append(
                    {
                        "budget": budget,
                        "multiplier": multiplier,
                        "ridge": ridge,
                        "fold": fold,
                        "component_mse": loss,
                    }
                )
                errors.append(loss)
            scores.append(
                {
                    "multiplier": multiplier,
                    "ridge": ridge,
                    "cv_mse": float(np.mean(errors)),
                }
            )
    best = min(scores, key=lambda d: d["cv_mse"])
    base.update(
        method="cv_scale",
        radii=(radius * best["multiplier"]).tolist(),
        cv_selection=best,
        selection_seconds=time.perf_counter() - start,
    )
    (ROOT / "results" / f"basis_cv_scale_{budget}.json").write_text(
        json.dumps(base, indent=2) + "\n"
    )
    print(budget, best)
pd.DataFrame(records).to_csv(ROOT / "results/scale_cv.csv", index=False)
(ROOT / "results/scale_cv_protocol.json").write_text(
    json.dumps(
        {
            "heldout_location_labels": labels.tolist(),
            "multipliers": [0.5, 1, 2],
            "ridge_relative_to_mean_gram_diagonal": [0.0001, 0.01, 1],
            "intervals": 3,
            "all_copies_of_each_sensor_location_share_fold": True,
            "withheld_truth_used": False,
            "final_weights_initialized_to_zero": True,
        },
        indent=2,
    )
    + "\n"
)
