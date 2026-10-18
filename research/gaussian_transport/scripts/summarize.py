"""Summaries and paired field-seed intervals, conditional on specified experiments."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT.parent / "track_observability"
rng = np.random.default_rng(314159)


def paired(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    idx = rng.integers(0, len(a), (10000, len(a)))
    ratio = 1 - b[idx].mean(axis=1) / a[idx].mean(axis=1)
    diff = (b - a)[idx].mean(axis=1)
    return dict(
        n=len(a),
        reference_mean=float(a.mean()),
        alternative_mean=float(b.mean()),
        relative_reduction=float(1 - b.mean() / a.mean()),
        relative_reduction_ci95=np.quantile(ratio, [0.025, 0.975]).tolist(),
        difference_ci95=np.quantile(diff, [0.025, 0.975]).tolist(),
    )


result = {}
a = pd.read_csv(ROOT / "results/transport.csv")
for flow in a.flow.unique():
    p = a[a.flow == flow].pivot(index="seed", columns="kind", values="center_rmse")
    result["transport_" + flow + "_df_vs_independent"] = paired(
        p["independent"], p["df"]
    )
g = pd.read_csv(TRACK / "results/geometry.csv")
p = g.pivot(index="seed", columns="geometry", values="mse")
result["geometry_correct_vs_straight"] = paired(p.straight, p.correct)
result["geometry_reversed_vs_straight"] = paired(p.straight, p.reversed)
g.groupby("geometry").mean(numeric_only=True).to_csv(
    TRACK / "results/geometry_summary.csv"
)
d = pd.read_csv(TRACK / "results/dipole_robustness.csv")
for dim in [2, 3]:
    for noise in [0.0, 0.05]:
        p = d[(d.dim == dim) & (d.noise_fraction == noise)].pivot(
            index="seed", columns="kind", values="mse"
        )
        result[f"dipole_dim{dim}_noise{noise}_df_vs_independent"] = paired(
            p.independent, p.df
        )
d.groupby(["dim", "noise_fraction", "kind"]).mean(numeric_only=True).to_csv(
    TRACK / "results/dipole_summary.csv"
)
o = pd.read_csv(TRACK / "results/public_robustness.csv")
o["sse"] = o.mse * o.n_test
o["bse"] = o.mean_baseline_mse * o.n_test
p = o.groupby(["dataset", "seed", "kind"])[["sse", "bse", "n_test"]].sum()
p["relative_mse"] = p.sse / p.bse
p.to_csv(TRACK / "results/public_robustness_summary.csv")
f = pd.read_csv(ROOT / "results/fluid_support_matched.csv")
f.groupby(["design", "kind"])[
    ["mse", "relative_energy_mse", "mean_baseline_mse"]
].mean().to_csv(ROOT / "results/fluid_support_summary.csv")
# Frames are one sequence: report per-frame paired effects, no event-population CI.
p = f[f.kind == "df"].pivot(index=["frame", "seed"], columns="design", values="mse")
frame = p.groupby("frame").mean()
frame["reduction_replace8"] = 1 - frame.lines_plus8_volume / frame.lines
frame.to_csv(ROOT / "results/fluid_frame_summary.csv")
result["fluid_support_matched"] = {
    "mean_line_mse": float(p.lines.mean()),
    "mean_replace8_mse": float(p.lines_plus8_volume.mean()),
    "relative_reduction": float(1 - p.lines_plus8_volume.mean() / p.lines.mean()),
    "per_frame_reduction_range": [
        float(frame.reduction_replace8.min()),
        float(frame.reduction_replace8.max()),
    ],
    "scope": "Four selected frames of one simulation; five sampling/noise seeds each. No event-level CI.",
}
(ROOT / "results/extended_summary.json").write_text(json.dumps(result, indent=2) + "\n")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
means = f.groupby(["design", "kind"]).relative_energy_mse.mean().unstack()
means.rename(
    index={
        "lines": "4 lines",
        "lines_plus8_volume": "88 line + 8 volume",
        "volume": "96 volume",
    },
    columns={
        "df": "Divergence-free",
        "helmholtz": "Helmholtz",
        "independent": "Independent",
    },
).plot.bar(ax=ax[0], rot=15)
ax[0].set(
    ylabel="Velocity MSE / target energy",
    title="ScalarSyn: all observations inside smoke",
    xlabel="96-sample design",
)
g.groupby("geometry").mse.mean().reindex(
    ["correct", "underestimated", "straight", "reversed", "naive_velocity"]
).plot.bar(ax=ax[1], rot=20, color="#4b78b9")
ax[1].set(
    ylabel="Vector MSE",
    xlabel="Geometry assumption",
    title="Synthetic arc geometry stress test",
)
fig.tight_layout()
fig.savefig(ROOT / "figures/fluid_and_geometry.png", dpi=180)
fig.savefig(ROOT / "figures/fluid_and_geometry.pdf")
plt.close(fig)
print(json.dumps(result, indent=2))
