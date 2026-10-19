"""Refit selected linear surrogates and separate image, sensor and prior losses.

This diagnostic never opens held-out motion truth or cameras. It is not another
selection rule, and its coefficients never initialize the nonlinear runs.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from selection import wendland_blocks

z = np.load(ROOT / "results/autodiff_dictionary.npz")
x, h, jac = z["centers"], z["radii"], z["jacobian"]
obs = np.load(ROOT / "results/selection_observations.npz")
yimage = obs["image_target"] / np.sqrt(len(jac))
image = jac / np.sqrt(len(jac))
sensors = json.loads(
    (
        ROOT.parent / "gaussian_observability/results/image_sensors_visual16.json"
    ).read_text()
)
y = np.stack(
    [np.array(sensors[str(f)]["velocity"]).ravel() for f in range(3)], axis=1
) / np.sqrt(48)
sensor = wendland_blocks(obs["sensor_xyz"], x, h).transpose(0, 2, 1, 3).reshape(
    48, len(x), 3
) / np.sqrt(48)
rows = []
for method in ["image_adols", "joint_adols", "joint_rols"]:
    trace = json.loads((ROOT / "results" / f"{method}_trace.json").read_text())
    for budget in [32, 64, 128]:
        basis = json.loads(
            (ROOT / "results" / f"basis_{method}_{budget}.json").read_text()
        )
        ii = np.array(basis["candidate_indices"])
        a = image[:, ii].reshape(len(jac), -1)
        b = sensor[:, ii].reshape(48, -1)
        design, target = a, yimage
        if method != "image_adols":
            design = np.concatenate([a, b])
            target = np.concatenate([yimage, y])
        ridge = trace.get("coefficient_ridge", 0)
        if ridge:
            design = np.concatenate([design, np.sqrt(ridge) * np.eye(3 * budget)])
            target = np.concatenate([target, np.zeros((3 * budget, 3))])
        beta, _, rank, sv = np.linalg.lstsq(design, target, rcond=1e-12)
        image_loss = float(np.sum((a @ beta - yimage) ** 2))
        sensor_loss = float(np.sum((b @ beta - y) ** 2))
        penalty = float(ridge * np.sum(beta**2))
        objective = (
            image_loss + (sensor_loss if method != "image_adols" else 0) + penalty
        )
        recorded = trace["steps"][budget - 1]["residual_sse"]
        rows.append(
            {
                "method": method,
                "budget": budget,
                "image_sse": image_loss,
                "sensor_sse": sensor_loss,
                "coefficient_penalty": penalty,
                "objective": objective,
                "trace_objective": recorded,
                "refit_relative_difference": abs(objective - recorded)
                / max(recorded, 1e-15),
                "rank": rank,
                "condition": float(sv[0] / sv[-1]),
                "radius30_fraction": float(np.mean(h[ii] == 30)),
                "radius60_fraction": float(np.mean(h[ii] == 60)),
                "radius120_fraction": float(np.mean(h[ii] == 120)),
            }
        )
pd.DataFrame(rows).to_csv(ROOT / "results/surrogate_diagnostics.csv", index=False)
report = {
    "normalized_initial_image_sse": float(np.sum(yimage**2)),
    "normalized_initial_sensor_sse": float(np.sum(y**2)),
    "scope": "Local normalized squared-loss proxy; these are not the nonlinear optimizer losses or measurement-noise likelihoods.",
}
(ROOT / "results/surrogate_scales.json").write_text(json.dumps(report, indent=2) + "\n")
print(report)
print(pd.DataFrame(rows).to_string(index=False))
