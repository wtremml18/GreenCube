"""Reproduce the predeclared synthetic experiment, retaining every outcome."""

import sys, json, time, platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.fields import tracks, target_grid, random_field, blind_field, null_audit
from src.kernels import tune, folds, divergence


def main():
    start = time.time()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rows = []
    candidates = []
    q = target_grid()
    for seed in range(40):
        truth = random_field(q, seed)
        for n in [12, 24, 48]:
            x, g = tracks(n)
            y = random_field(x, seed) + np.random.default_rng(1000 + seed).normal(
                0, 0.05, x.shape
            )
            for cv in ["random", "track"]:
                splits = folds(len(x), g if cv == "track" else None, seed + 4000)
                for kind in ["df", "independent"]:
                    model, best, scores = tune(x, y, splits, kind)
                    pred = model.predict(q)
                    mse = np.mean(np.sum((pred - truth) ** 2, axis=1))
                    rows.append(
                        dict(
                            seed=seed,
                            n=len(x),
                            cv=cv,
                            kind=kind,
                            mse=mse,
                            divergence_rms=np.sqrt(
                                np.mean(divergence(model, q[::8]) ** 2)
                            ),
                            **best,
                        )
                    )
                    candidates.extend(
                        dict(seed=seed, n=len(x), cv=cv, kind=kind, **v) for v in scores
                    )
        pd.DataFrame(rows).to_csv(out / "synthetic.csv", index=False)
        if seed % 5 == 0:
            print(f"seed {seed}/39; {time.time() - start:.1f} s", flush=True)
    pd.DataFrame(candidates).to_csv(out / "synthetic_candidates.csv", index=False)
    (out / "synthetic_run.json").write_text(
        json.dumps(
            dict(
                seconds=time.time() - start,
                python=platform.python_version(),
                numpy=np.__version__,
                seeds=40,
            ),
            indent=2,
        )
    )
    # Counterexample uses identical data and model for all amplitudes and signs.
    x, g = tracks(24)
    y = random_field(x, 0)
    model, best, _ = tune(x, y, folds(len(x), seed=4000), "df")
    base = random_field(q, 0)
    b = blind_field(q)
    b /= np.sqrt(np.mean(np.sum(b * b, axis=1)))
    pred = model.predict(q)
    blind = []
    for a in [0, 0.25, 0.5, 1, 2]:
        for sign in [-1, 1]:
            blind.append(
                dict(
                    amplitude=a,
                    sign=sign,
                    mse=np.mean(np.sum((pred - base - sign * a * b) ** 2, axis=1)),
                    cv_mse=best["cv_mse"],
                )
            )
    pd.DataFrame(blind).to_csv(out / "blind.csv", index=False)
    np.savez_compressed(
        out / "blind_fields.npz", q=q, base=base, blind=b, pred=pred, x=x, y=y
    )
    audit = []
    for cutoff in [3, 4, 5]:
        for n in [12, 24, 48, 96]:
            x, _ = tracks(n)
            a = null_audit(x, q, cutoff)
            audit.append(
                dict(
                    design="parallel",
                    n=len(x),
                    cutoff=cutoff,
                    jitter=0.0,
                    **{k: v for k, v in a.items() if k != "coefficients"},
                )
            )
        x, _ = tracks(24)
        for jitter in [0.0001, 0.001, 0.01]:
            z = x.copy()
            z[:, 1] += np.random.default_rng(81).normal(0, jitter, len(z))
            a = null_audit(z, q, cutoff)
            audit.append(
                dict(
                    design="jittered",
                    n=len(z),
                    cutoff=cutoff,
                    jitter=jitter,
                    **{k: v for k, v in a.items() if k != "coefficients"},
                )
            )
        for extra in [4, 8, 16, 32]:
            z = np.vstack(
                [
                    x,
                    np.random.default_rng(82).uniform(
                        [0.025, 0.125], [0.975, 0.875], (extra, 2)
                    ),
                ]
            )
            a = null_audit(z, q, cutoff)
            audit.append(
                dict(
                    design="off_track",
                    n=len(z),
                    cutoff=cutoff,
                    jitter=0.0,
                    **{k: v for k, v in a.items() if k != "coefficients"},
                )
            )
    pd.DataFrame(audit).to_csv(out / "observability.csv", index=False)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
