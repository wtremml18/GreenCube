"""Density-conditioned sensor placement on released ScalarSyn volumes.

Volumetric upper-bound study, NOT camera-only Gaussian-splat reconstruction.
"""

import argparse, json, sys, time, gzip
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "track_observability"))
from design import condition, scores, farthest
from src.kernels import kernel


def vector_kernel(x, z, length, kind):
    return 4 * kernel(x, z, length, kind) + np.tile(4 * np.eye(3), (len(x), len(z)))


def posterior(q, x, normal, labels, length, kind, sigma):
    kqq = vector_kernel(q, q, length, kind)
    if labels is None:
        return np.zeros(3 * len(q)), kqq
    kxx = vector_kernel(x, x, length, kind).reshape(len(x), 3, len(x), 3)
    obs = np.einsum("ia,iajb,jb->ij", normal, kxx, normal)
    obs.flat[:: len(obs) + 1] += sigma**2
    cross = np.einsum(
        "aib,ib->ai",
        vector_kernel(q, x, length, kind).reshape(3 * len(q), len(x), 3),
        normal,
    )
    fac = cho_factor(obs, lower=True)
    mean = cross @ cho_solve(fac, labels)
    cov = kqq - cross @ cho_solve(fac, cross.T)
    return mean, (cov + cov.T) / 2


def volume(folder, kind, frame):
    a = np.load(folder / f"{kind}_{frame:04d}.npz")["arr_0"][:100, :150, :100].astype(
        float
    )
    return a[..., 0] if kind == "density" else a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--length", type=float, default=0.16)
    p.add_argument("--density-noise", type=float, default=0.3)
    p.add_argument("--kind", choices=["df", "independent"], default="df")
    p.add_argument("--tag", default="primary")
    p.add_argument("--sensor-sigma", type=float, default=0.05)
    a = p.parse_args()
    rows = []
    audit = []
    designs = []
    begin = time.time()
    for frame in [1, 10, 20, 28]:
        rho = volume(a.data, "density", frame)
        vel = volume(a.data, "velocity", frame)
        # Voxel coordinates: vector components x,y,z; array axes z,y,x.
        # Upstream ScalarSyn config: dt = sim_steps / scale = 2 / 2 = 1.
        # Central difference spans two video-frame intervals.
        temporal = (
            volume(a.data, "density", frame + 1) - volume(a.data, "density", frame - 1)
        ) / 2
        grad = np.stack(np.gradient(rho, edge_order=2)[::-1], axis=-1)
        norm = np.linalg.norm(grad, axis=-1)
        support = rho > 0.01 * rho.max()
        support[:5] = support[-5:] = False
        support[:, :15] = support[:, -5:] = False
        support[:, :, :5] = support[:, :, -5:] = False
        pool = np.argwhere(support)
        good = norm > np.quantile(norm[support], 0.3)
        for seed in range(a.seeds):
            rng = np.random.default_rng(93000 + frame * 100 + seed)
            perm = rng.permutation(len(pool))
            test = pool[perm[:256]]
            candidates = pool[perm[256:448]]
            leftover = pool[perm[448:]]
            obs = leftover[good[tuple(leftover.T)]][:256]
            n = grad[tuple(obs.T)] / norm[tuple(obs.T)][:, None]
            label = -temporal[tuple(obs.T)] / norm[tuple(obs.T)]
            oracle = np.sum(vel[tuple(obs.T)] * n, axis=1)
            q = np.vstack([candidates, test])[:, ::-1] / 100
            x = obs[:, ::-1] / 100
            truth = vel[tuple(test.T)].reshape(-1)
            candidate_values = vel[tuple(candidates.T)] + rng.normal(
                0, 0.05, (len(candidates), 3)
            )
            # Candidate/test/optical observations are disjoint original voxels.
            assert len(np.unique(np.vstack([obs, candidates, test]), axis=0)) == len(
                obs
            ) + len(candidates) + len(test)
            audit.append(
                dict(
                    frame=frame,
                    seed=seed,
                    normal_rmse=float(np.sqrt(np.mean((label - oracle) ** 2))),
                    normal_truth_rms=float(np.sqrt(np.mean(oracle**2))),
                    mean_gradient=float(norm[tuple(obs.T)].mean()),
                )
            )
            target_indices = np.arange(3 * len(candidates), 3 * len(q))
            random_order = rng.permutation(len(candidates))
            for optical, labels in [
                ("none", None),
                ("density", label),
                ("oracle_normal", oracle),
            ]:
                base_mean, base_cov = posterior(
                    q, x, n, labels, a.length, a.kind, a.density_noise
                )
                for design in ["random", "farthest", "variance", "integrated"]:
                    mean = base_mean.copy()
                    cov = base_cov.copy()
                    selected = []
                    for budget in range(17):
                        if budget in [0, 2, 4, 8, 16]:
                            pred = mean[target_indices]
                            var = np.diag(cov)[target_indices]
                            err = pred - truth
                            rows.append(
                                dict(
                                    frame=frame,
                                    seed=seed,
                                    optical=optical,
                                    design=design,
                                    budget=budget,
                                    kind=a.kind,
                                    length=a.length,
                                    density_noise=a.density_noise,
                                    mse=float(np.mean(err**2)),
                                    vector_rmse=float(np.sqrt(3 * np.mean(err**2))),
                                    coverage95=float(
                                        np.mean(
                                            np.abs(err)
                                            <= 1.96 * np.sqrt(np.maximum(var, 0))
                                        )
                                    ),
                                    posterior_variance=float(var.mean()),
                                )
                            )
                        if budget == 16:
                            break
                        available = np.setdiff1d(np.arange(len(candidates)), selected)
                        if design == "random":
                            chosen = int(random_order[budget])
                        elif design == "farthest":
                            chosen = farthest(q[: len(candidates)], selected)
                        else:
                            score = scores(
                                cov,
                                available,
                                target_indices,
                                a.sensor_sigma**2,
                                "integrated" if design == "integrated" else "variance",
                            )
                            chosen = int(available[np.argmax(score)])
                        selected.append(chosen)
                        ids = 3 * chosen + np.arange(3)
                        mean, cov = condition(
                            mean, cov, ids, candidate_values[chosen], a.sensor_sigma**2
                        )
                    designs.append(
                        dict(
                            frame=frame,
                            seed=seed,
                            optical=optical,
                            design=design,
                            selected_zyx=candidates[selected].tolist(),
                            test_zyx=test.tolist(),
                            density_zyx=obs.tolist(),
                        )
                    )
            pd.DataFrame(rows).to_csv(
                ROOT / "results" / f"design_{a.tag}.csv", index=False
            )
            print(frame, seed, "elapsed", round(time.time() - begin, 1), flush=True)
    pd.DataFrame(audit).to_csv(
        ROOT / "results" / f"density_audit_{a.tag}.csv", index=False
    )
    (ROOT / "results" / f"selections_{a.tag}.json.gz").write_bytes(
        gzip.compress(json.dumps(designs).encode(), mtime=0)
    )
    (ROOT / "results" / f"config_{a.tag}.json").write_text(
        json.dumps(
            dict(
                seeds=a.seeds,
                length=a.length,
                kind=a.kind,
                density_noise=a.density_noise,
                frames=[1, 10, 20, 28],
                sensor_noise_actual=0.05,
                sensor_noise_assumed=a.sensor_sigma,
                n_candidates=192,
                n_test=256,
                n_density=256,
                effective_frame_dt=1.0,
                runtime_seconds=time.time() - begin,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
