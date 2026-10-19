"""Evaluate saved models on the unchanged common motion targets from prior work."""

import argparse
import json
import sys
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

samples = np.load(ROOT.parent / "gaussian_observability/results/common_samples.npz")
rows = []
preds = {}
for path in sorted((ROOT / "results").glob("run_*.json")):
    d = json.loads(path.read_text())
    e = json.loads((ROOT / "results" / f"eval_{d['tag']}.json").read_text())
    for frame in range(3):
        idx = samples[f"indices_frame{frame}"]
        truth = samples[f"truth_frame{frame}"]
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
        assert model.n_kernels == d["budget"]
        with torch.no_grad():
            prediction = (
                model(
                    torch.tensor(
                        idx[:, ::-1].copy(), dtype=torch.float32, device="cuda"
                    )
                )
                .cpu()
                .numpy()
            )
        preds[f"{d['tag']}_frame{frame}"] = prediction
        rows.append(
            dict(
                **d,
                frame=frame,
                vector_rmse=float(
                    np.sqrt(np.mean(np.sum((prediction - truth) ** 2, axis=1)))
                ),
                n=len(truth),
                reserved_psnr=e["images"]["test"]["psnr"],
                train_psnr=e["images"]["train"]["psnr"],
            )
        )
np.savez_compressed(ROOT / "results/common_predictions.npz", **preds)
pd.DataFrame(rows).to_csv(ROOT / "results/common_metrics.csv", index=False)
print(
    pd.DataFrame(rows)
    .groupby(["sensor_weight", "epochs", "budget", "method"])[
        ["vector_rmse", "reserved_psnr", "train_wall_seconds"]
    ]
    .mean()
    .to_string()
)
