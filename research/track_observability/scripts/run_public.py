"""Exploratory nested spatial holdout of two preprocessed ocean datasets."""

import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.kernels import folds, tune


def quadrants(x):
    mid = np.median(x, axis=0)
    return (x[:, 0] >= mid[0]).astype(int) + 2 * (x[:, 1] >= mid[1]).astype(int)


def main():
    rows = []
    splits_saved = {}
    predictions = []
    candidates = []
    start = time.time()
    for dataset in ["gladfull", "laserfull"]:
        xy = np.loadtxt(ROOT / "data" / f"XY_train_{dataset}.csv", delimiter=",")
        uv = np.loadtxt(ROOT / "data" / f"UV_train_{dataset}.csv", delimiter=",")
        # Random fixed subset, chosen before inspecting velocity labels.
        selected = np.sort(
            np.random.default_rng(20261018).choice(
                len(xy), min(160, len(xy)), replace=False
            )
        )
        x = xy[selected]
        y = uv[selected]
        for fold, (tr, te) in enumerate(folds(len(x), quadrants(x))):
            # Isotropic coordinate scaling preserves physical divergence constraints.
            origin = x[tr].mean(axis=0)
            scale = np.ptp(x[tr], axis=0).max()
            xx = (x - origin) / scale
            inner = folds(len(tr), quadrants(xx[tr]))
            splits_saved[f"{dataset}_{fold}"] = dict(
                original_rows=selected.tolist(),
                train=tr.tolist(),
                test=te.tolist(),
                origin=origin.tolist(),
                scale=float(scale),
                inner=[dict(train=a.tolist(), test=b.tolist()) for a, b in inner],
            )
            norm = np.mean(np.sum((y[te] - y[tr].mean(axis=0)) ** 2, axis=1))
            for kind in ["df", "independent", "helmholtz"]:
                model, best, scores = tune(xx[tr], y[tr], inner, kind)
                p = model.predict(xx[te])
                mse = np.mean(np.sum((p - y[te]) ** 2, axis=1))
                rows.append(
                    dict(
                        dataset=dataset,
                        fold=fold,
                        kind=kind,
                        n_train=len(tr),
                        n_test=len(te),
                        mse=mse,
                        mean_baseline_mse=norm,
                        relative_mse=mse / norm,
                        **best,
                    )
                )
                candidates.extend(
                    dict(dataset=dataset, fold=fold, kind=kind, **v) for v in scores
                )
                predictions.extend(
                    dict(
                        dataset=dataset,
                        fold=fold,
                        kind=kind,
                        row=int(selected[i]),
                        u=float(y[i, 0]),
                        v=float(y[i, 1]),
                        u_pred=float(a[0]),
                        v_pred=float(a[1]),
                    )
                    for i, a in zip(te, p)
                )
            print(dataset, fold, "seconds", round(time.time() - start, 1), flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/public.csv", index=False)
    pd.DataFrame(candidates).to_csv(ROOT / "results/public_candidates.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        ROOT / "results/public_predictions.csv", index=False
    )
    (ROOT / "results/public_splits.json").write_text(json.dumps(splits_saved, indent=2))


if __name__ == "__main__":
    main()
