"""Summarize every completed run, charge selection cost, plot fixed-budget curves."""

import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "results/common_metrics.csv")
protocol = json.loads((ROOT / "results/selection_protocol.json").read_text())
costs = {}
for method in ["image_ols", "joint_ols", "image_adols", "joint_adols", "joint_rols"]:
    method_protocol = (
        protocol
        if method in {"image_ols", "joint_ols"}
        else json.loads((ROOT / "results/selection_protocol_autodiff.json").read_text())
    )
    costs[method] = (
        method_protocol["jacobian_seconds"]
        + json.loads((ROOT / "results" / f"{method}_trace.json").read_text())[
            "selection_seconds"
        ]
    )
rows = []
for tag, d in df.groupby("tag"):
    r = d.iloc[0]
    # Report the upstream logged compute time separately from initialization/I/O.
    log = (ROOT / "logs" / f"{tag}_train.log").read_text()
    times = re.findall(r"Training time: (\d+)h (\d+)m ([\d.]+)s", log)
    assert times, f"Missing optimization timing for {tag}"
    compute = sum(3600 * int(h) + 60 * int(m) + float(s) for h, m, s in times)
    selection = costs.get(r.method, 0)
    rows.append(
        {
            "tag": tag,
            "method": r.method,
            "stage": "finite_difference_pilot"
            if r.method in {"image_ols", "joint_ols"}
            else "backward_selector_or_control",
            "budget": r.budget,
            "seed": r.seed,
            "sensor_weight": r.sensor_weight,
            "epochs": r.epochs,
            "vector_rmse": d.vector_rmse.mean(),
            "reserved_psnr": r.reserved_psnr,
            "train_psnr": r.train_psnr,
            "train_wall_seconds": r.train_wall_seconds,
            "optimization_seconds": compute,
            "selection_seconds": selection,
            "selection_plus_train_seconds": selection + r.train_wall_seconds,
            "motion_parameters": 3 * 7 * r.budget,
        }
    )
run = pd.DataFrame(rows)
run.to_csv(ROOT / "results/run_summary.csv", index=False)
keys = ["budget", "seed", "sensor_weight", "epochs"]
base = run[run.method == "poisson"][keys + ["vector_rmse", "reserved_psnr"]].rename(
    columns={"vector_rmse": "baseline_rmse", "reserved_psnr": "baseline_psnr"}
)
paired = run[run.method != "poisson"].merge(base, on=keys, validate="many_to_one")
paired["motion_change_percent"] = 100 * (paired.vector_rmse / paired.baseline_rmse - 1)
paired["reserved_psnr_change_db"] = paired.reserved_psnr - paired.baseline_psnr
paired.to_csv(ROOT / "results/paired_comparisons.csv", index=False)
summary = (
    run.groupby(["sensor_weight", "epochs", "budget", "method"])
    .agg(
        n=("seed", "count"),
        vector_rmse=("vector_rmse", "mean"),
        min_rmse=("vector_rmse", "min"),
        max_rmse=("vector_rmse", "max"),
        reserved_psnr=("reserved_psnr", "mean"),
        train_wall_seconds=("train_wall_seconds", "mean"),
        optimization_seconds=("optimization_seconds", "mean"),
        selection_plus_train_seconds=("selection_plus_train_seconds", "mean"),
    )
    .reset_index()
)
summary.to_csv(ROOT / "results/summary.csv", index=False)
reference = pd.read_csv(
    ROOT.parent / "gaussian_observability/results/image_summary.csv"
)
reference = reference[reference.condition.isin(["visual16", "images-visual16"])]
reference.groupby("condition").agg(
    vector_rmse=("vector_rmse", "mean"),
    reserved_psnr=("all_frame_test_psnr", "mean"),
    seeds=("seed", "nunique"),
).to_csv(ROOT / "results/reference_summary.csv")
full = reference[reference.condition == "visual16"]
labels = {
    "poisson": "Domain coverage, broad radius",
    "coverage": "Image-support coverage, mixed scales",
    "random": "Random dictionary blocks",
    "image_adols": "Image OLS (backward)",
    "joint_adols": "Image + sensor OLS (backward)",
    "joint_rols": "Regularized image + sensor OLS",
}
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
sub = summary[(summary.sensor_weight == 1) & (summary.epochs == 200)]
for method, label in labels.items():
    s = sub[sub.method == method].sort_values("budget")
    for ax, column in zip(
        axes, ["vector_rmse", "reserved_psnr", "selection_plus_train_seconds"]
    ):
        (line,) = ax.plot(s.budget, s[column], "o-", label=label)
        if column == "vector_rmse" and (s.n > 1).any():
            ax.fill_between(
                s.budget, s.min_rmse, s.max_rmse, alpha=0.12, color=line.get_color()
            )
axes[0].axhline(
    full.vector_rmse.mean(), color="gray", ls="--", label="Earlier 1,121-kernel model"
)
axes[1].axhline(full.all_frame_test_psnr.mean(), color="gray", ls="--")
for ax in axes:
    ax.set_xlabel("Motion RBFs per interval")
    ax.set_xticks([32, 64, 128])
    ax.grid(alpha=0.2)
axes[0].set_ylabel("Mean vector RMSE (native units)")
axes[1].set_ylabel("Reserved-camera PSNR (dB)")
axes[2].set_ylabel("Selection + training wall time (s)")
axes[0].legend(fontsize=7)
fig.suptitle(
    "One scene, fixed inputs; 3 seeds for domain/coverage/joint OLS/ROLS; 1 otherwise"
)
fig.tight_layout()
fig.savefig(ROOT / "figures/budget_curves.png", dpi=180)
fig.savefig(ROOT / "figures/budget_curves.pdf")
plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for method in ["image_adols", "joint_adols", "joint_rols"]:
    trace = json.loads((ROOT / "results" / f"{method}_trace.json").read_text())["steps"]
    axes[0].semilogy(
        [r["step"] for r in trace],
        [r["residual_sse"] for r in trace],
        label=labels[method],
    )
    d = json.loads((ROOT / "results" / f"basis_{method}_128.json").read_text())
    r = np.array(d["radii"])
    axes[1].plot(
        [32, 64, 128],
        [np.mean(r[:b] == 30) for b in [32, 64, 128]],
        "o-",
        label=labels[method],
    )
axes[0].set(xlabel="Selected blocks", ylabel="Selection objective (penalized for ROLS)")
axes[1].set(xlabel="Selected blocks", ylabel="Fraction with radius 30", ylim=(0, 1))
for ax in axes:
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
fig.suptitle("A decreasing selection loss does not certify physical motion")
fig.tight_layout()
fig.savefig(ROOT / "figures/selection.png", dpi=180)
fig.savefig(ROOT / "figures/selection.pdf")
print(summary.to_string(index=False))
