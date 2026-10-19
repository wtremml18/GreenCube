"""Does the regularized selector need the image residual/Jacobian?

Keep image-derived candidate geometry and the same sensor-selected ridge. Remove
only image rows from block selection. This is a follow-up, not a new training run.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from selection import block_ols, wendland_blocks

start = time.perf_counter()
z = np.load(ROOT / "results/autodiff_dictionary.npz")
sensors = json.loads(
    (
        ROOT.parent / "gaussian_observability/results/image_sensors_visual16.json"
    ).read_text()
)
x = np.array(sensors["0"]["xyz"])
centers = z["centers"]
radii = z["radii"]
a = wendland_blocks(x, centers, radii).transpose(0, 2, 1, 3).reshape(
    48, len(centers), 3
) / np.sqrt(48)
y = np.stack(
    [np.array(sensors[str(f)]["velocity"]).ravel() for f in range(3)], axis=1
) / np.sqrt(48)
reference = json.loads((ROOT / "results/joint_rols_trace.json").read_text())
ridge = reference["coefficient_ridge"]
features = np.concatenate(
    [a, np.sqrt(ridge) * np.eye(3 * len(centers)).reshape(-1, len(centers), 3)]
)
values = np.concatenate([y, np.zeros((3 * len(centers), 3))])
order, trace = block_ols(features, values, 128)
comparison = []
for budget in [32, 64, 128]:
    b = json.loads((ROOT / "results" / f"basis_joint_rols_{budget}.json").read_text())
    ii = np.array(order[:budget])
    joint = b["candidate_indices"]
    comparison.append(
        {
            "budget": budget,
            "same_order": ii.tolist() == joint,
            "same_set": set(ii) == set(joint),
            "overlap": len(set(ii) & set(joint)),
        }
    )
    (ROOT / "results" / f"basis_sensor_rols_{budget}.json").write_text(
        json.dumps(
            {
                "method": "sensor_rols",
                "budget": budget,
                "centers": centers[ii].tolist(),
                "radii": radii[ii].tolist(),
                "candidate_indices": ii.tolist(),
            },
            indent=2,
        )
        + "\n"
    )
report = {
    "selection_seconds": time.perf_counter() - start,
    "comparisons": comparison,
    "scope": "Image-derived candidate positions retained; image residual and renderer Jacobian removed. No new reconstruction fits counted.",
    "steps": trace,
}
(ROOT / "results/image_selection_ablation.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(comparison)
