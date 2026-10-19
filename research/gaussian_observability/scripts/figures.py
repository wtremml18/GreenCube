"""Figures from released images and saved measured errors; no generated imagery."""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--upstream", type=Path, required=True)
a = p.parse_args()
base = json.loads((ROOT / "results/images_images-visual16_s42_w1.json").read_text())
added = json.loads((ROOT / "results/images_visual16_s42_w1.json").read_text())
bdir = (
    a.upstream
    / "log/greencube"
    / base["run_name"]
    / "window_0_4/evaluation/images/test"
)
sdir = (
    a.upstream
    / "log/greencube"
    / added["run_name"]
    / "window_0_4/evaluation/images/test"
)
fig, ax = plt.subplots(2, 3, figsize=(9, 5.5))
for row, frame in enumerate([0, 3]):
    paths = [
        bdir / f"cam_{frame:03d}_frame_{frame:03d}_gt.png",
        bdir / f"cam_{frame:03d}_frame_{frame:03d}_render.png",
        sdir / f"cam_{frame:03d}_frame_{frame:03d}_render.png",
    ]
    gt = np.asarray(Image.open(paths[0]))
    mask = gt[..., :3].mean(-1) > 0.01 * gt[..., :3].mean(-1).max()
    yy, xx = np.where(mask)
    y0, y1 = max(0, yy.min() - 8), min(len(gt), yy.max() + 9)
    x0, x1 = max(0, xx.min() - 8), min(gt.shape[1], xx.max() + 9)
    for col, path in enumerate(paths):
        ax[row, col].imshow(np.asarray(Image.open(path))[y0:y1, x0:x1])
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
        if row == 0:
            ax[row, col].set_title(
                [
                    "Reserved camera image",
                    "Images + physical constraints",
                    "Plus 16 sensor locations",
                ][col],
                fontsize=10,
            )
    ax[row, 0].set_ylabel(f"Frame {frame}")
fig.suptitle("Public ScalarSyn: reserved view, cropped to smoke (seed 42)")
fig.tight_layout()
fig.savefig(ROOT / "figures/camera_comparison.png", dpi=180)
fig.savefig(ROOT / "figures/camera_comparison.pdf")
plt.close(fig)
df = pd.read_csv(ROOT / "results/image_summary.csv")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for seed in [42, 43, 44]:
    sub = df[(df.seed == seed) & (df.weight == 1)]
    b = sub[sub.condition == "images-visual16"]
    s = sub[sub.condition == "visual16"]
    axes[0].plot(
        [0, 1], [b.vector_rmse.mean(), s.vector_rmse.mean()], "o-", label=f"Seed {seed}"
    )
    axes[1].plot(
        [0, 1], [b.all_frame_test_psnr.mean(), s.all_frame_test_psnr.mean()], "o-"
    )
for axis in axes:
    axis.set_xticks([0, 1], ["Images + physics", "Plus 16 sensors"])
    axis.grid(alpha=0.2)
axes[0].set(
    ylabel="Vector RMSE (native velocity units)", title="Unobserved smoke voxels"
)
sensor_only = pd.read_csv(ROOT / "results/sensor_only.csv")
reference = sensor_only[
    (sensor_only.layout == "image_sensors_visual16") & (sensor_only.kind == "df")
].vector_rmse.mean()
axes[0].axhline(reference, color="black", linestyle="--", label="Sensor-only DF RBF")
axes[0].legend()
axes[1].set(ylabel="PSNR (dB)", title="Reserved camera, all four frames")
fig.suptitle("Three optimization seeds, shared initial geometry, one public scene")
fig.tight_layout()
fig.savefig(ROOT / "figures/image_results.png", dpi=180)
fig.savefig(ROOT / "figures/image_results.pdf")
plt.close(fig)
# Released density is used here only for a post-selection spatial-support audit.
rho = np.load(a.upstream / "gt/scalarsyn/density_0000.npz")["arr_0"][
    :100, :150, :100, 0
]
cloud = np.argwhere(rho > 0.01 * rho.max())
rng = np.random.default_rng(82)
cloud = cloud[rng.choice(len(cloud), 3000, replace=False)]
fig = plt.figure(figsize=(9, 5))
for j, (filename, title) in enumerate(
    [
        ("image_sensors16.json", "Opacity-only placement"),
        ("image_sensors_visual16.json", "Four-camera silhouette placement"),
    ]
):
    ax = fig.add_subplot(1, 2, j + 1, projection="3d")
    ax.scatter(cloud[:, 2], cloud[:, 0], cloud[:, 1], c="gray", alpha=0.025, s=2)
    loc = np.array(json.loads((ROOT / "results" / filename).read_text())["0"]["xyz"])
    ax.scatter(loc[:, 0], loc[:, 2], loc[:, 1], c="#e14938", s=25, depthshade=False)
    ax.set(
        xlim=(0, 100),
        ylim=(0, 100),
        zlim=(0, 150),
        xlabel="x",
        ylabel="z",
        zlabel="y",
        title=title,
    )
fig.suptitle("16 fixed sensor locations; gray = public smoke support (audit only)")
fig.tight_layout()
fig.savefig(ROOT / "figures/sensor_support.png", dpi=180)
fig.savefig(ROOT / "figures/sensor_support.pdf")
plt.close(fig)
