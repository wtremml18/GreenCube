"""Describe learned centers/radii and timing without using motion truth."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--upstream", type=Path, required=True)
a = p.parse_args()
rows = []
for path in sorted((ROOT / "results").glob("run_*.json")):
    r = json.loads(path.read_text())
    basis = json.loads((ROOT / "results" / r["basis_file"]).read_text())
    c = np.array(basis["centers"])
    h = np.array(basis["radii"])
    for frame in range(3):
        state = torch.load(
            a.upstream
            / "log/greencube"
            / r["run_name"]
            / "frame_velocities"
            / f"frame_{frame:03d}_velocity.pth",
            map_location="cpu",
            weights_only=False,
        )
        centers = state["init_centers"].numpy()
        radii = state["init_radii"].numpy()
        rows.append(
            {
                "tag": r["tag"],
                "method": r["method"],
                "budget": r["budget"],
                "seed": r["seed"],
                "sensor_weight": r["sensor_weight"],
                "epochs": r["epochs"],
                "frame": frame,
                "median_initial_radius": float(np.median(h)),
                "median_final_radius": float(np.median(radii)),
                "mean_abs_radius_change": float(np.mean(np.abs(radii - h))),
                "rms_center_displacement": float(
                    np.sqrt(np.mean(np.sum((centers - c) ** 2, axis=1)))
                ),
                "motion_parameter_count": 7 * len(h),
            }
        )
pd.DataFrame(rows).to_csv(ROOT / "results/learned_basis.csv", index=False)
print("Audited", len(rows), "learned velocity bases")

reference = []
for r in json.loads(
    (
        ROOT.parent / "gaussian_observability/results/image_runs_visual16_w1.json"
    ).read_text()
):
    for frame in range(3):
        state = torch.load(
            a.upstream
            / "log/greencube"
            / r["run_name"]
            / "frame_velocities"
            / f"frame_{frame:03d}_velocity.pth",
            map_location="cpu",
            weights_only=False,
        )
        reference.append(
            {
                "condition": r["condition"],
                "seed": r["seed"],
                "frame": frame,
                "kernels": state["n_kernels"],
                "run_name": r["run_name"],
            }
        )
pd.DataFrame(reference).to_csv(ROOT / "results/reference_basis.csv", index=False)
assert all(r["kernels"] == 1121 for r in reference)
