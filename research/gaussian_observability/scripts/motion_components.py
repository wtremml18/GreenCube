"""Post-fit diagnostic of errors normal/tangent to released density contours.

GT density is used only to interpret held-out errors, never to modify a fit.
The tangent subspace has two dimensions; the normal has one. This is not a
proof of a temporal rendering nullspace or a deployable fusion estimator.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
a = p.parse_args()
samples = np.load(ROOT / "results/common_samples.npz")
camera = np.load(ROOT / "results/common_motion_predictions.npz")
sensor = np.load(ROOT / "results/sensor_only_predictions.npz")
normals = {}
for frame in range(9):
    rho = np.load(a.data / f"density_{frame:04d}.npz")["arr_0"][:100, :150, :100, 0]
    grad = np.stack(np.gradient(rho), axis=-1)[..., ::-1]
    idx = samples[f"indices_frame{frame}"]
    g = grad[tuple(idx.T)]
    mag = np.linalg.norm(g, axis=1)
    keep = (mag > np.quantile(mag, 0.3)) & (mag > 1e-8)
    normals[frame] = (keep, g[keep] / mag[keep, None])


def diagnostic(prediction, frame):
    keep, n = normals[frame]
    truth = samples[f"truth_frame{frame}"][keep]
    err = prediction[keep] - truth
    normal2 = np.sum(err * n, axis=1) ** 2
    total2 = np.sum(err**2, axis=1)
    tangent2 = np.maximum(total2 - normal2, 0)
    return dict(
        n=int(keep.sum()),
        normal_mse=float(normal2.mean()),
        tangent_vector_mse=float(tangent2.mean()),
        tangent_per_dimension_mse=float(tangent2.mean() / 2),
        total_vector_mse=float(total2.mean()),
        truth_normal_mse=float(np.mean(np.sum(truth * n, axis=1) ** 2)),
        truth_vector_mse=float(np.mean(np.sum(truth**2, axis=1))),
    )


rows = []
for r in pd.read_csv(ROOT / "results/common_motion.csv").itertuples():
    key = f"{r.condition}_s{r.seed}_w{r.weight:g}_frame{r.frame}"
    rows.append(
        dict(
            method=r.condition,
            seed=r.seed,
            weight=r.weight,
            frame=r.frame,
            **diagnostic(camera[key], r.frame),
        )
    )
for r in pd.read_csv(ROOT / "results/sensor_only.csv").itertuples():
    key = f"{r.layout}_{r.kind}_frame{r.frame}"
    rows.append(
        dict(
            method=f"{r.layout}_{r.kind}",
            seed=-1,
            weight=0,
            frame=r.frame,
            **diagnostic(sensor[key], r.frame),
        )
    )
out = pd.DataFrame(rows)
assert np.allclose(out.normal_mse + out.tangent_vector_mse, out.total_vector_mse)
out.to_csv(ROOT / "results/motion_components.csv", index=False)
chosen = out[
    out.method.isin(
        [
            "images-visual16",
            "visual16",
            "image_sensors_visual16_df",
            "images-visual16-f10",
            "visual16-f10",
            "image_sensors_visual16_f10_df",
        ]
    )
]
print(
    chosen.groupby("method")[["normal_mse", "tangent_vector_mse", "total_vector_mse"]]
    .mean()
    .to_string()
)
