"""Freeze evaluation locations shared by camera and sensor-only reconstructions."""

import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
a = p.parse_args()
layouts = [
    json.loads(p.read_text()) for p in (ROOT / "results").glob("image_sensors*.json")
]
locations = np.unique(np.vstack([np.array(d["0"]["xyz"]) for d in layouts]), axis=0)
out = {}
for frame in range(9):
    rho = np.load(a.data / f"density_{frame:04d}.npz")["arr_0"][:100, :150, :100, 0]
    gt = np.load(a.data / f"velocity_{frame:04d}.npz")["arr_0"][:100, :150, :100]
    support = rho > 0.01 * rho.max()
    support[:5] = support[-5:] = False
    support[:, :20] = support[:, -5:] = False
    support[:, :, :5] = support[:, :, -5:] = False
    pool = np.argwhere(support)
    far = np.min(np.sum((pool[:, None, ::-1] - locations) ** 2, axis=-1), axis=1) > 4
    pool = pool[far]
    idx = pool[
        np.random.default_rng(7400 + frame).choice(len(pool), 4096, replace=False)
    ]
    out[f"indices_frame{frame}"] = idx
    out[f"truth_frame{frame}"] = gt[tuple(idx.T)]
np.savez_compressed(ROOT / "results/common_samples.npz", **out)
(ROOT / "results/common_sample_protocol.json").write_text(
    json.dumps(
        dict(
            test_seed_base=7400,
            n_test_per_frame=4096,
            exclusion_distance_voxels=2,
            all_sensor_xyz=locations.tolist(),
            sensor_layout_files=sorted(
                p.name for p in (ROOT / "results").glob("image_sensors*.json")
            ),
        ),
        indent=2,
    )
    + "\n"
)
print("Frozen 9 x 4096 held-out voxels; exclusion locations:", len(locations))
