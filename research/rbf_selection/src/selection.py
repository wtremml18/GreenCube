"""Block OLS for a linearized observation map, with three columns per center.

Established greedy regression: not a global optimality guarantee.
"""

import numpy as np


def wendland_blocks(points, centers, radii):
    """Upstream's 3D divergence-free Wendland C4 normalization, (n,c,3,3)."""
    x = (np.asarray(points)[:, None] - np.asarray(centers)[None]) / np.asarray(radii)[
        None, :, None
    ]
    r = np.linalg.norm(x, axis=-1)
    t = np.maximum(1 - r, 0) ** 4
    c1 = 30 * t
    c2 = (1 + 4 * r - 5 * r * r) * t
    return (2 * c2 - c1 * r * r)[..., None, None] * np.eye(3) + c1[..., None, None] * x[
        ..., :, None
    ] * x[..., None, :]


def block_ols(matrix, target, budget, rank_tolerance=1e-8):
    """Choose whole vector-kernel blocks by residual least-squares improvement.

    matrix: observations x candidates x 3, target: observations x outputs.
    Maintains orthogonality with a second projection to suppress roundoff drift.
    """
    a = np.asarray(matrix, dtype=np.float64).copy()
    y = np.asarray(target, dtype=np.float64).copy()
    if y.ndim == 1:
        y = y[:, None]
    n, candidates, _block = a.shape
    if not 0 < budget <= candidates:
        raise ValueError("Budget must be positive and no larger than dictionary")
    selected, trace, q_all = [], [], np.empty((n, 0))
    initial_energy = np.sum(a * a, axis=(0, 2))
    scale = max(initial_energy.max(), 1e-30)
    for step in range(budget):
        gram = np.einsum("nci,ncj->cij", a, a, optimize=True)
        value, vec = np.linalg.eigh(gram)
        keep = value > np.maximum(value[:, -1:] * rank_tolerance, scale * 1e-14)
        inv = np.zeros_like(value)
        np.divide(1, value, out=inv, where=keep)
        cross = np.einsum("nci,no->cio", a, y, optimize=True)
        projected = np.einsum("cik,cio->cko", vec, cross)
        gains = np.sum(projected**2 * inv[..., None], axis=(1, 2))
        gains[selected] = -np.inf
        chosen = int(np.argmax(gains))
        if not np.isfinite(gains[chosen]):
            raise ValueError("Exhausted dictionary")
        u = a[:, chosen] @ vec[chosen][:, keep[chosen]]
        if u.shape[1]:
            u /= np.sqrt(value[chosen][keep[chosen]])
            if q_all.shape[1]:
                u -= q_all @ (q_all.T @ u)
            u, s, _ = np.linalg.svd(u, full_matrices=False)
            u = u[:, s > 1e-7]
            flat = a.reshape(n, -1)
            flat -= u @ (u.T @ flat)
            y -= u @ (u.T @ y)
            q_all = np.column_stack((q_all, u))
        selected.append(chosen)
        trace.append(
            {
                "step": step + 1,
                "candidate": chosen,
                "predicted_gain": float(gains[chosen]),
                "residual_sse": float(np.sum(y * y)),
                "rank": int(q_all.shape[1]),
            }
        )
    return selected, trace


def farthest_order(points, count):
    points = np.asarray(points)
    selected = [int(np.argmin(np.sum((points - points.mean(0)) ** 2, axis=1)))]
    distance = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for _ in range(1, count):
        distance[selected] = -np.inf
        i = int(np.argmax(distance))
        selected.append(i)
        distance = np.minimum(distance, np.sum((points - points[i]) ** 2, axis=1))
    return np.array(selected)
