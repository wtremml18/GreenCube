"""Direct sensor regression in the same feature family; no image loss/truth tuning."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from selection import farthest_order, wendland_blocks

sensors = json.loads(
    (
        ROOT.parent / "gaussian_observability/results/image_sensors_visual16.json"
    ).read_text()
)
x = np.array(sensors["0"]["xyz"])
y = np.stack([np.array(sensors[str(f)]["velocity"]).ravel() for f in range(3)], axis=1)
z = np.load(ROOT / "results/autodiff_dictionary.npz")
common = np.load(ROOT.parent / "gaussian_observability/results/common_samples.npz")
selection = json.loads((ROOT / "results/joint_rols_trace.json").read_text())
full_penalty = 48 * selection["coefficient_ridge"]
conditions = [("full_bank_sensor", z["centers"], z["radii"], full_penalty)]
for n in [32, 64, 128]:
    b = json.loads((ROOT / "results" / f"basis_joint_rols_{n}.json").read_text())
    conditions.append(
        (
            "selected_rols_fixed_ridge",
            np.array(b["centers"]),
            np.array(b["radii"]),
            full_penalty,
        )
    )
    b = json.loads((ROOT / "results" / f"basis_cv_scale_{n}.json").read_text())
    centers = np.array(b["centers"])
    radii = np.array(b["radii"])
    a = wendland_blocks(x, centers, radii).transpose(0, 2, 1, 3).reshape(48, -1)
    penalty = b["cv_selection"]["ridge"] * np.trace(a @ a.T) / 48
    conditions.append(("domain_cv_sensor", centers, radii, penalty))
labels = np.empty(16, dtype=int)
labels[np.random.default_rng(713).permutation(16)] = np.arange(16) % 4
cv_records = []
for seed in [914, 915, 916]:
    pool = np.random.default_rng(seed).uniform(
        [5, 20, 5], [95, 145, 95], size=(4096, 3)
    )
    center_pool = pool[farthest_order(pool, 192)]
    centers = np.repeat(center_pool, 3, axis=0)
    radii = np.tile([30.0, 60.0, 120.0], 192)
    matrix = wendland_blocks(x, centers, radii).transpose(0, 2, 1, 3).reshape(48, -1)
    scores = []
    for alpha in np.logspace(-6, 2, 9):
        losses = []
        for fold in range(4):
            tr = np.repeat(labels != fold, 3)
            va = np.repeat(labels == fold, 3)
            a = matrix[tr]
            gram = a @ a.T
            lam = alpha * np.trace(gram) / len(gram)
            weight = a.T @ np.linalg.solve(gram + lam * np.eye(len(gram)), y[tr])
            loss = float(np.mean((matrix[va] @ weight - y[va]) ** 2))
            cv_records.append(
                {
                    "seed": seed,
                    "relative_ridge": alpha,
                    "fold": fold,
                    "component_mse": loss,
                }
            )
            losses.append(loss)
        scores.append((float(np.mean(losses)), alpha))
    _, alpha = min(scores)
    penalty = alpha * np.trace(matrix @ matrix.T) / 48
    kind = f"domain_bank_s{seed}"
    conditions.append((kind, centers, radii, penalty))
    (ROOT / "results" / f"basis_{kind}_576.json").write_text(
        json.dumps(
            {
                "method": kind,
                "budget": 576,
                "centers": centers.tolist(),
                "radii": radii.tolist(),
                "relative_ridge": alpha,
            },
            indent=2,
        )
        + "\n"
    )
pd.DataFrame(cv_records).to_csv(ROOT / "results/domain_bank_cv.csv", index=False)

rows = []
predictions = {}
for kind, centers, radii, penalty in conditions:
    a = wendland_blocks(x, centers, radii).transpose(0, 2, 1, 3).reshape(48, -1)
    weights = (a.T @ np.linalg.solve(a @ a.T + penalty * np.eye(48), y)).reshape(
        len(centers), 3, 3
    )
    for frame in range(3):
        xyz = common[f"indices_frame{frame}"][:, ::-1]
        pieces = []
        for start in range(0, len(xyz), 512):
            b = wendland_blocks(xyz[start : start + 512], centers, radii)
            pieces.append(np.einsum("ncij,cj->ni", b, weights[:, :, frame]))
        prediction = np.concatenate(pieces).astype(np.float32)
        key = f"{kind}_n{len(centers)}_frame{frame}"
        predictions[key] = prediction
        truth = common[f"truth_frame{frame}"]
        rows.append(
            {
                "kind": kind,
                "budget": len(centers),
                "frame": frame,
                "vector_rmse": float(
                    np.sqrt(np.mean(np.sum((prediction - truth) ** 2, axis=1)))
                ),
                "coefficient_ridge": penalty,
                "n": len(xyz),
                "prediction_key": key,
            }
        )
np.savez_compressed(ROOT / "results/linear_sensor_only_predictions.npz", **predictions)
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results/linear_sensor_only.csv", index=False)
summary = df.groupby(["kind", "budget"]).vector_rmse.mean().reset_index()
summary.to_csv(ROOT / "results/linear_sensor_only_summary.csv", index=False)
print(summary.to_string(index=False))
