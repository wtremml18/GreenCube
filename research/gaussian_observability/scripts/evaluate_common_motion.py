"""Re-evaluate every completed camera model on one common held-out motion sample.

All layouts share 4096 test voxels per frame, separated by >2 voxels from every
simulated instrument in every tested layout. Image scores are not modified.
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--upstream", type=Path, required=True)
a = p.parse_args()
a.upstream = a.upstream.resolve()
sys.path.insert(0, str(a.upstream))
from velocity_common.dfrbf import TiDFRBF

files = sorted((ROOT / "results").glob("images_*_s*_w*.json"))
records = [(f, json.loads(f.read_text())) for f in files]
sample_data = np.load(ROOT / "results/common_samples.npz")
protocol = json.loads((ROOT / "results/common_sample_protocol.json").read_text())
locations = np.array(protocol["all_sensor_xyz"])
samples = {
    frame: (sample_data[f"indices_frame{frame}"], sample_data[f"truth_frame{frame}"])
    for frame in range(9)
}
rows = []
cache = {}
outputs = {}
all_data = {}
for path, d in records:
    name = path.stem.removeprefix("images_")
    condition, rest = name.split("_s", 1)
    seed, weight = rest.split("_w", 1)
    for original in d["velocity"]:
        frame = original["frame"]
        idx, truth = samples[frame]
        key = (d["run_name"], frame)
        if key not in cache:
            model = TiDFRBF.load(
                str(
                    a.upstream
                    / "log/greencube"
                    / d["run_name"]
                    / "frame_velocities"
                    / f"frame_{frame:03d}_velocity.pth"
                ),
                device="cuda",
            )
            with torch.no_grad():
                pred = (
                    model(
                        torch.tensor(
                            idx[:, ::-1].copy(), dtype=torch.float32, device="cuda"
                        )
                    )
                    .cpu()
                    .numpy()
                )
            cache[key] = pred
        pred = cache[key]
        rows.append(
            dict(
                condition=condition,
                seed=int(seed),
                weight=float(weight),
                frame=frame,
                vector_rmse=float(
                    np.sqrt(np.mean(np.sum((pred - truth) ** 2, axis=1)))
                ),
                zero_rmse=float(np.sqrt(np.mean(np.sum(truth**2, axis=1)))),
                n=len(idx),
                run_name=d["run_name"],
            )
        )
        all_data[f"{name}_frame{frame}"] = pred
for frame, (idx, truth) in samples.items():
    all_data[f"indices_frame{frame}"] = idx
    all_data[f"truth_frame{frame}"] = truth
np.savez_compressed(ROOT / "results/common_motion_predictions.npz", **all_data)
pd.DataFrame(rows).to_csv(ROOT / "results/common_motion.csv", index=False)
(ROOT / "results/common_motion_protocol.json").write_text(
    json.dumps(
        dict(
            test_seed_base=7400,
            n_test_per_frame=4096,
            exclusion_distance_voxels=2,
            all_sensor_xyz=locations.tolist(),
            n_completed_evaluations=len(records),
            n_rows=len(rows),
            native_velocity_units=True,
        ),
        indent=2,
    )
    + "\n"
)
print(
    "Evaluated",
    len(records),
    "camera conditions on common samples:",
    len(rows),
    "frame rows",
)
