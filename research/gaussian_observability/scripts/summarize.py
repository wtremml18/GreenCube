"""Summarize all declared sensitivity runs; never select a winning setting."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
tags = ["primary", "short", "long", "discrepancy", "independent"]
frames = []
for tag in tags:
    a = pd.read_csv(ROOT / "results" / f"design_{tag}.csv")
    a["setting"] = tag
    frames.append(a)
df = pd.concat(frames, ignore_index=True)
summary = (
    df.groupby(["setting", "optical", "design", "budget"])[
        ["mse", "coverage95", "posterior_variance"]
    ]
    .mean()
    .reset_index()
)
summary.to_csv(ROOT / "results/design_summary.csv", index=False)
# Monte Carlo sampling variability conditional on this one simulation and its four frames.
rng = np.random.default_rng(20261018)
comparisons = []
for budget in [8, 16]:
    sub = df[
        (df.setting == "primary") & (df.optical == "density") & (df.budget == budget)
    ]
    perseed = sub.groupby(["seed", "design"]).mse.mean().unstack()
    for baseline in ["random", "farthest", "variance"]:
        ratios = []
        for _ in range(5000):
            ids = rng.integers(0, len(perseed), len(perseed))
            sample = perseed.iloc[ids]
            ratios.append(1 - sample.integrated.mean() / sample[baseline].mean())
        comparisons.append(
            dict(
                budget=budget,
                baseline=baseline,
                relative_mse_reduction=float(
                    1 - perseed.integrated.mean() / perseed[baseline].mean()
                ),
                conditional_seed_interval95=np.quantile(
                    ratios, [0.025, 0.975]
                ).tolist(),
            )
        )
(ROOT / "results/design_comparisons.json").write_text(
    json.dumps(comparisons, indent=2) + "\n"
)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
labels = {
    "random": "Random",
    "farthest": "Spatial coverage",
    "variance": "Point variance",
    "integrated": "Integrated variance",
}
for design, label in labels.items():
    d = summary[
        (summary.setting == "primary")
        & (summary.optical == "density")
        & (summary.design == design)
    ]
    axes[0].plot(d.budget, d.mse, "o-", label=label)
    axes[1].plot(d.budget, d.coverage95, "o-")
    axes[2].plot(d.budget, d.posterior_variance, "o-")
axes[0].set(title="Actual velocity error", ylabel="Component MSE")
axes[1].set(
    title="Marginal interval coverage",
    ylabel="Coverage of nominal 95% intervals",
    ylim=(0.55, 1),
)
axes[1].axhline(0.95, color="black", linestyle="--", lw=1)
axes[2].set(title="Predicted uncertainty", ylabel="Mean posterior component variance")
for ax in axes:
    ax.set_xlabel("Vector measurements")
    ax.set_xticks([0, 4, 8, 16])
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=8)
fig.suptitle(
    "Public ScalarSyn density + sparse velocities: error and confidence disagree"
)
fig.tight_layout()
fig.savefig(ROOT / "figures/design.png", dpi=180)
fig.savefig(ROOT / "figures/design.pdf")
plt.close(fig)
# Include every successful camera run and all frames; repeated baselines are identifiable.
rows = []
for path in sorted((ROOT / "results").glob("images_*_s*_w*.json")):
    d = json.loads(path.read_text())
    tag = path.stem.removeprefix("images_")
    condition, rest = tag.split("_s", 1)
    seed, weight = rest.split("_w", 1)
    psnr = d["images"]["test"]["per_camera"]["psnr"]
    ids = d["images"]["test"]["per_camera"]["frame_indices"]
    for v in d["velocity"]:
        rows.append(
            dict(
                condition=condition,
                seed=int(seed),
                weight=float(weight),
                frame=v["frame"],
                vector_rmse=v["vector_rmse"],
                zero_rmse=v["zero_rmse"],
                test_psnr=float(
                    np.mean([p for p, f in zip(psnr, ids) if f == v["frame"]])
                ),
                all_frame_test_psnr=d["images"]["test"]["psnr"],
                run_name=d["run_name"],
            )
        )
if rows:
    im = pd.DataFrame(rows)
    common_path = ROOT / "results/common_motion.csv"
    if common_path.exists():
        common = pd.read_csv(common_path)
        im = im.rename(columns={"vector_rmse": "original_layout_sample_rmse"})
        im = im.merge(
            common[["condition", "seed", "weight", "frame", "vector_rmse"]],
            on=["condition", "seed", "weight", "frame"],
            validate="one_to_one",
        )
        assert len(im) == len(rows)
    im.to_csv(ROOT / "results/image_summary.csv", index=False)
    print(
        im.groupby(["weight", "condition"])[["vector_rmse", "all_frame_test_psnr"]]
        .mean()
        .to_string()
    )
print(json.dumps(comparisons, indent=2))
