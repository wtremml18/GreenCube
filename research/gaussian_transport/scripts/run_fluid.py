"""Sparse direct-velocity reconstruction on public ScalarSyn ground truth."""

import sys, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "track_observability"))
import numpy as np
import pandas as pd
from src.kernels import tune, folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-matched", action="store_true")
    args = parser.parse_args()
    suffix = "_support_matched" if args.support_matched else ""
    rows = []
    ambiguity = []
    samples = []
    for frame in [0, 10, 20, 29]:
        rho = np.load(ROOT / "data" / f"density_{frame:04d}.npz")["arr_0"][
            :100, :150, :100, 0
        ]
        vel = np.load(ROOT / "data" / f"velocity_{frame:04d}.npz")["arr_0"][
            :100, :150, :100
        ]
        # Array axes z,y,x; vector components x,y,z. Uniform grid spacing.
        gz, gy, gx = np.gradient(rho, 0.01, edge_order=2)
        grad = np.stack([gx, gy, gz], axis=-1)
        tangent = np.cross(grad, np.array([0.3, -0.4, 0.5]))
        tangent /= np.sqrt(np.mean(np.sum(tangent * tangent, axis=-1)))
        dot = np.sum(tangent * grad, axis=-1)
        div = (
            np.gradient(tangent[..., 0], 0.01, axis=2, edge_order=2)
            + np.gradient(tangent[..., 1], 0.01, axis=1, edge_order=2)
            + np.gradient(tangent[..., 2], 0.01, axis=0, edge_order=2)
        )
        inner = np.s_[2:-2, 2:-2, 2:-2]
        vdiv = (
            np.gradient(vel[..., 0], 0.01, axis=2, edge_order=2)
            + np.gradient(vel[..., 1], 0.01, axis=1, edge_order=2)
            + np.gradient(vel[..., 2], 0.01, axis=0, edge_order=2)
        )
        ambiguity.append(
            dict(
                frame=frame,
                tangent_rms=float(np.sqrt(np.mean(np.sum(tangent * tangent, axis=-1)))),
                density_derivative_change_rms=float(np.sqrt(np.mean(dot[inner] ** 2))),
                tangent_divergence_rms=float(np.sqrt(np.mean(div[inner] ** 2))),
                source_finite_difference_divergence_rms=float(
                    np.sqrt(np.mean(vdiv[inner] ** 2))
                ),
            )
        )
        mask = rho > 0.01 * rho.max()
        mask[:5] = False
        mask[-5:] = False
        mask[:, :5] = False
        mask[:, -5:] = False
        mask[:, :, :5] = False
        mask[:, :, -5:] = False
        pool = np.argwhere(mask)
        line = np.array(
            [
                [z, y, x]
                for z in [30, 70]
                for x in [30, 70]
                for y in np.round(np.linspace(15, 135, 24)).astype(int)
            ]
        )
        if args.support_matched:
            line = []
            for z in [45, 55]:
                for x in [45, 55]:
                    ys = np.flatnonzero(mask[z, :, x])
                    if len(ys) < 24:
                        raise ValueError("Insufficient support on fixed control lines")
                    line.extend(
                        [
                            [z, int(y), x]
                            for y in ys[
                                np.round(np.linspace(0, len(ys) - 1, 24)).astype(int)
                            ]
                        ]
                    )
            line = np.array(line)
        # All target voxels exclude any possible line observations.
        line_ids = np.ravel_multi_index(line.T, rho.shape)
        pool_ids = np.ravel_multi_index(pool.T, rho.shape)
        pool = pool[~np.isin(pool_ids, line_ids)]
        for seed in range(5):
            rng = np.random.default_rng(80000 + 100 * frame + seed)
            order = rng.permutation(len(pool))
            test = pool[order[:512]]
            volume = pool[order[512:608]]
            # Randomly replace 8 line entries; preserve total count.
            keep = np.sort(rng.choice(96, 88, replace=False))
            designs = {
                "lines": line,
                "lines_plus8_volume": np.vstack([line[keep], volume[:8]]),
                "volume": volume,
            }
            truth = vel[tuple(test.T)]
            q = test[:, ::-1] / 100
            noise = rng.normal(size=(96, 3))
            for design, train in designs.items():
                x = train[:, ::-1] / 100
                clean = vel[tuple(train.T)]
                sigma = (
                    0.01 * np.sqrt(np.mean(vel[mask] ** 2))
                    if args.support_matched
                    else 0.01 * np.sqrt(np.mean(clean**2))
                )
                y = clean + sigma * noise
                samples.append(
                    dict(
                        frame=frame,
                        seed=seed,
                        design=design,
                        train_zyx=train.tolist(),
                        test_zyx=test.tolist(),
                        noise_sigma=float(sigma),
                    )
                )
                baseline = np.mean(np.sum((truth - clean.mean(axis=0)) ** 2, axis=1))
                energy = np.mean(np.sum(truth**2, axis=1))
                for kind in ["df", "independent", "helmholtz"]:
                    m, b, _ = tune(
                        x,
                        y,
                        folds(len(x), seed=seed),
                        kind,
                        lengths=(0.08, 0.16, 0.32, 0.64),
                        ridges=(0.001, 0.03, 0.3),
                    )
                    mse = np.mean(np.sum((m.predict(q) - truth) ** 2, axis=1))
                    rows.append(
                        dict(
                            frame=frame,
                            seed=seed,
                            design=design,
                            kind=kind,
                            mse=mse,
                            relative_energy_mse=mse / energy,
                            mean_baseline_mse=baseline,
                            training_smoke_fraction=float(
                                np.mean(mask[tuple(train.T)])
                            ),
                            **b,
                        )
                    )
            print("fluid", frame, seed, flush=True)
            pd.DataFrame(rows).to_csv(
                ROOT / "results" / f"fluid{suffix}.csv", index=False
            )
        pd.DataFrame(ambiguity).to_csv(
            ROOT / "results/density_ambiguity.csv", index=False
        )
    (ROOT / "results" / f"fluid_samples{suffix}.json").write_text(json.dumps(samples))


if __name__ == "__main__":
    main()
