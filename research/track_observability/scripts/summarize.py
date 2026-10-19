"""Generate all reported summaries and publication figures from raw result files."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
F = ROOT / "figures"
F.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
rng = np.random.default_rng(314159)
idx = rng.integers(0, 40, size=(10000, 40))


def ci(a):
    a = np.asarray(a)
    m = a[idx].mean(axis=1)
    return [float(a.mean()), *np.quantile(m, [0.025, 0.975]).tolist()]


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    delta = ci(a - b)
    red = 1 - a[idx].mean(axis=1) / b[idx].mean(axis=1)
    return dict(
        difference=delta,
        relative_reduction=[
            float(1 - a.mean() / b.mean()),
            *np.quantile(red, [0.025, 0.975]).tolist(),
        ],
    )


d = pd.read_csv(R / "synthetic.csv")
s = []
for (n, cv, kind), g in d.groupby(["n", "cv", "kind"]):
    a = ci(g.sort_values("seed").mse)
    s.append(
        dict(
            n=n,
            cv=cv,
            kind=kind,
            mean_mse=a[0],
            ci_low=a[1],
            ci_high=a[2],
            mean_cv_mse=g.cv_mse.mean(),
            divergence_rms=g.divergence_rms.mean(),
        )
    )
s = pd.DataFrame(s)
s.to_csv(R / "synthetic_summary.csv", index=False)


def get(n, cv, kind):
    return (
        d.query("n==@n and cv==@cv and kind==@kind").sort_values("seed").mse.to_numpy()
    )


summary = {
    "df_vs_independent_96": paired(
        get(96, "random", "df"), get(96, "random", "independent")
    ),
    "track_vs_random_96": paired(get(96, "track", "df"), get(96, "random", "df")),
    "density_192_vs_48": paired(get(192, "random", "df"), get(48, "random", "df")),
}
a = pd.read_csv(R / "design.csv")
for n in [8, 32]:
    off = a.query('extra==@n and design=="off"').sort_values("seed").mse
    along = a.query('extra==@n and design=="along"').sort_values("seed").mse
    summary[f"off_vs_along_{n}"] = paired(off, along)
a.groupby(["extra", "design"]).mse.agg(["mean", "std"]).to_csv(R / "design_summary.csv")
p = pd.read_csv(R / "public.csv")
public = []
for (dataset, kind), g in p.groupby(["dataset", "kind"]):
    mse = np.average(g.mse, weights=g.n_test)
    baseline = np.average(g.mean_baseline_mse, weights=g.n_test)
    public.append(
        dict(
            dataset=dataset,
            kind=kind,
            mse=mse,
            mean_baseline_mse=baseline,
            relative_mse=mse / baseline,
        )
    )
pub = pd.DataFrame(public)
pub.to_csv(R / "public_summary.csv", index=False)
(R / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
# 1: identical observations, very different truths.
b = np.load(R / "blind_fields.npz")
q = b["q"]
shape = (25, 31)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
for ax, v, title in zip(
    axs,
    [b["base"], b["base"] + b["blind"], b["blind"]],
    ["Base field", "Base + invisible perturbation", "Invisible perturbation"],
):
    im = ax.pcolormesh(
        q[:, 0].reshape(shape),
        q[:, 1].reshape(shape),
        np.linalg.norm(v, axis=1).reshape(shape),
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=2.8,
    )
    keep = np.arange(len(q)).reshape(shape)[::3, ::3].ravel()
    ax.quiver(
        q[keep, 0],
        q[keep, 1],
        v[keep, 0],
        v[keep, 1],
        color="white",
        scale=20,
        width=0.004,
    )
    for y in [0.125, 0.375, 0.625, 0.875]:
        ax.axhline(y, color="#d95f02", lw=0.9, ls="--")
    ax.set(xlabel="x", ylabel="y", title=title, xlim=(0, 1), ylim=(0.1, 0.9))
    ax.set_aspect("equal")
fig.colorbar(im, ax=axs, label="Vector magnitude", shrink=0.75)
fig.savefig(F / "blind_field.pdf")
fig.savefig(F / "blind_field.png", dpi=180)
plt.close(fig)
# 2: CV vs actual, and geometry intervention.
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
colors = {"df": "#0072B2", "independent": "#D55E00"}
for kind in colors:
    g = s.query('cv=="random" and kind==@kind')
    axs[0].plot(g.n, g.mean_mse, "o-", color=colors[kind], label=f"{kind}: grid error")
    axs[0].plot(g.n, g.mean_cv_mse, "s--", color=colors[kind], label=f"{kind}: CV loss")
    axs[0].fill_between(g.n, g.ci_low, g.ci_high, color=colors[kind], alpha=0.12)
axs[0].set(
    xlabel="Observations on four fixed tracks",
    ylabel="Vector MSE",
    yscale="log",
    title="(a) Along-track density",
)
axs[0].legend(fontsize=7)
for cv, style in [("random", "o-"), ("track", "s--")]:
    g = s.query('kind=="df" and cv==@cv')
    axs[1].plot(g.n, g.mean_mse, style, label=cv)
    axs[1].fill_between(g.n, g.ci_low, g.ci_high, alpha=0.15)
axs[1].set(
    xlabel="Observations",
    ylabel="Clean-grid vector MSE",
    title="(b) DF tuning strategy",
)
axs[1].legend()
for design in ["along", "off"]:
    means = [get(96, "random", "df").mean()]
    los = [ci(get(96, "random", "df"))[1]]
    his = [ci(get(96, "random", "df"))[2]]
    for n in [8, 32]:
        c = ci(a.query("extra==@n and design==@design").sort_values("seed").mse)
        means.append(c[0])
        los.append(c[1])
        his.append(c[2])
    axs[2].plot([0, 8, 32], means, "o-", label=design)
    axs[2].fill_between([0, 8, 32], los, his, alpha=0.15)
axs[2].set(
    xlabel="Additional observations",
    ylabel="Clean-grid vector MSE",
    title="(c) Exploratory intervention",
)
axs[2].legend()
fig.savefig(F / "validation.pdf")
fig.savefig(F / "validation.png", dpi=180)
plt.close(fig)
# 3: exact vs noisy radius and basis sensitivity.
e = pd.read_csv(R / "observability.csv")
v = pd.read_csv(R / "noise_observability.csv")
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
for cutoff in [3, 4, 5]:
    g = e.query('design=="parallel" and cutoff==@cutoff')
    axs[0].plot(g.n, g.radius, "o-", label=f"cutoff {cutoff}")
axs[0].set(
    xlabel="Along-track observations",
    ylabel="Exact-null RMS radius",
    title="(a) Persistent ambiguity",
)
axs[0].legend(fontsize=8)
g = e.query('design=="jittered" and cutoff==4')
axs[1].plot(g.jitter, g.radius, "k--o", label="Exact null")
for delta in [0.001, 0.01, 0.05]:
    g = v.query('design=="jittered" and cutoff==4 and delta==@delta')
    axs[1].plot(g.jitter, g.radius, "o-", label=f"budget {delta}")
axs[1].set(
    xlabel="Transverse jitter SD",
    ylabel="RMS radius",
    xscale="log",
    title="(b) Rank versus stability",
)
axs[1].legend(fontsize=7)
for cutoff in [3, 4, 5]:
    g = v.query('design=="off_track" and cutoff==@cutoff and delta==.01')
    axs[2].plot(g.n - 96, g.radius, "o-", label=f"cutoff {cutoff}")
axs[2].set(
    xlabel="Additional off-track observations",
    ylabel="Noise-budget RMS radius",
    title="(c) Geometry and model class",
)
axs[2].legend(fontsize=8)
fig.savefig(F / "observability.pdf")
fig.savefig(F / "observability.png", dpi=180)
plt.close(fig)
# 4: all public folds, with pooled MSE relative to training-mean baseline.
fig, axs = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
for ax, dataset in zip(axs, ["gladfull", "laserfull"]):
    for i, kind in enumerate(["df", "independent", "helmholtz"]):
        g = p.query("dataset==@dataset and kind==@kind")
        h = pub.query("dataset==@dataset and kind==@kind")
        ax.scatter(np.full(len(g), i), g.relative_mse, color="gray", s=20)
        ax.plot(i, h.relative_mse.iloc[0], "D", color="#0072B2", ms=7)
    ax.axhline(1, color="black", ls="--", lw=0.8)
    ax.set(
        xticks=range(3),
        xticklabels=["DF", "Independent", "Helmholtz"],
        ylabel="MSE / training-mean MSE",
        title=dataset.replace("full", "").upper(),
    )
fig.savefig(F / "public_holdout.pdf")
fig.savefig(F / "public_holdout.png", dpi=180)
plt.close(fig)
print(json.dumps(summary, indent=2))
print(pub.to_string(index=False))
