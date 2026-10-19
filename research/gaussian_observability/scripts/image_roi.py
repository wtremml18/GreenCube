"""Evaluate saved 8-bit renders in a smoke ROI, avoiding a black-background score."""

from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--upstream", type=Path, required=True)
a = p.parse_args()
rows = []
for path in sorted((ROOT / "results").glob("images_*_s*_w*.json")):
    d = json.loads(path.read_text())
    name = path.stem.removeprefix("images_")
    condition, rest = name.split("_s", 1)
    seed, weight = rest.split("_w", 1)
    frames = d["images"]["test"]["per_camera"]["frame_indices"]
    n = max(frames) + 1
    directory = (
        a.upstream
        / "log/greencube"
        / d["run_name"]
        / f"window_0_{n}/evaluation/images/test"
    )
    for frame in frames:
        gt = (
            np.asarray(
                Image.open(directory / f"cam_{frame:03d}_frame_{frame:03d}_gt.png")
            ).astype(float)
            / 255
        )
        pred = (
            np.asarray(
                Image.open(directory / f"cam_{frame:03d}_frame_{frame:03d}_render.png")
            ).astype(float)
            / 255
        )
        intensity = gt[..., :3].mean(-1)
        mask = intensity > 0.01 * intensity.max()
        yy, xx = np.where(mask)
        y0 = max(0, yy.min() - 8)
        y1 = min(len(gt), yy.max() + 9)
        x0 = max(0, xx.min() - 8)
        x1 = min(gt.shape[1], xx.max() + 9)
        target = gt[y0:y1, x0:x1, :3]
        estimate = pred[y0:y1, x0:x1, :3]
        mse = np.mean((estimate - target) ** 2)
        rows.append(
            dict(
                condition=condition,
                seed=int(seed),
                weight=float(weight),
                frame=int(frame),
                roi_psnr=float(-10 * np.log10(mse)),
                roi_relative_rmse=float(
                    np.sqrt(np.sum((estimate - target) ** 2) / np.sum(target**2))
                ),
                roi_bounds_xyxy=[int(x0), int(y0), int(x1), int(y1)],
                roi_fraction=float(
                    target.shape[0] * target.shape[1] / (gt.shape[0] * gt.shape[1])
                ),
            )
        )
pd.DataFrame(rows).to_csv(ROOT / "results/image_roi.csv", index=False)
print("Verified foreground ROI scores for", len(rows), "camera frames")
