"""An exact full-time counterexample, plus integration convergence."""

import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ambiguity import generator, flow_map, sensor_operator
from transport import advect, flow

s = np.array([0.6, 0.35, 0.2])
sigma = np.diag(s * s)
omega = np.array([0.3, 0.5, 1.0])
a = generator(s, omega)
rng = np.random.default_rng(97)
markers = rng.normal(size=(200, 3)) * s
rows = []
for time in [0.0, 0.25, 0.5, 1.0, 2.0]:
    f = flow_map(s, omega, time)
    moved = markers @ f.T
    cov = f @ sigma @ f.T
    # Gaussian covariance identical => full density and every line projection identical.
    rows.append(
        dict(
            time=time,
            covariance_max_abs=float(np.max(np.abs(cov - sigma))),
            logdet=float(np.linalg.slogdet(f)[1]),
            marker_rms_displacement=float(
                np.sqrt(np.mean(np.sum((moved - markers) ** 2, axis=1)))
            ),
        )
    )
sensors = {}
for name, p in {
    "center": [[0, 0, 0]],
    "one_point": [[0.2, 0.1, 0.05]],
    "two_noncollinear": [[0.2, 0.1, 0.05], [-0.1, 0.15, 0.1]],
}.items():
    op = sensor_operator(np.array(p), s)
    sv = np.linalg.svd(op, compute_uv=False)
    sensors[name] = dict(
        rank=int(np.linalg.matrix_rank(op)), singular_values=sv.tolist()
    )
# Numerical ODE convergence, independent of density counterexample.
c = rng.uniform(-0.4, 0.4, (8, 3))
cov = np.broadcast_to(np.diag([0.02**2, 0.03**2, 0.04**2]), (8, 3, 3)).copy()
reference = advect(flow, c, cov, duration=2.0, steps=256)
convergence = []
for steps in [4, 8, 16, 32, 64]:
    m, cc, f = advect(flow, c, cov, duration=2.0, steps=steps)
    convergence.append(
        dict(
            steps=steps,
            center_max_error=float(np.max(np.abs(m - reference[0]))),
            covariance_max_error=float(np.max(np.abs(cc - reference[1]))),
            max_abs_logdet=float(np.max(np.abs(np.linalg.slogdet(f)[1]))),
        )
    )
result = dict(
    generator_trace=float(np.trace(a)),
    strain_norm=float(np.linalg.norm((a + a.T) / 2)),
    time_series=rows,
    sensors=sensors,
    integration_convergence=convergence,
)
(ROOT / "results/exact_ambiguity.json").write_text(json.dumps(result, indent=2) + "\n")
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
f = flow_map(s, omega, 1.0)
moved = markers @ f.T
for ax, pts, title in zip(
    axes,
    [markers, moved],
    ["Material markers at t=0", "Same density, moved markers at t=1"],
):
    ax.scatter(pts[:, 0], pts[:, 1], c=np.arange(len(pts)), s=10, cmap="viridis")
    ax.set(xlim=(-1.8, 1.8), ylim=(-1.1, 1.1), xlabel="x", ylabel="y", title=title)
    angle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(s[0] * np.cos(angle), s[1] * np.sin(angle), "k--", alpha=0.5)
fig.tight_layout()
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures/density_ambiguity.png", dpi=180)
fig.savefig(ROOT / "figures/density_ambiguity.pdf")
plt.close(fig)
print(json.dumps(result, indent=2))
