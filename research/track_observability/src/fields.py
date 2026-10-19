"""Analytic solenoidal fields and finite-dimensional observability audits."""

import numpy as np
from scipy.linalg import svd


def wavevectors(cutoff):
    return np.array(
        [
            (i, j)
            for i in range(cutoff + 1)
            for j in range(-cutoff, cutoff + 1)
            if i > 0 or j > 0
        ],
        dtype=float,
    )


def basis(x, cutoff=4, decay=1.5):
    """Rotated gradients of weighted Fourier sin/cos potentials; point-major rows.

    ||coefficients||_2 is the explicitly chosen potential coefficient budget.
    Constant velocity is omitted: this audit concerns spatial perturbations.
    """
    x = np.asarray(x)
    k = wavevectors(cutoff)
    phase = 2 * np.pi * x @ k.T
    rotation = 2 * np.pi * np.column_stack((k[:, 1], -k[:, 0]))
    rotation *= (1 + np.sum(k * k, axis=1))[:, None] ** (-decay)
    blocks = np.concatenate(
        (np.cos(phase)[:, :, None] * rotation, -np.sin(phase)[:, :, None] * rotation),
        axis=1,
    )
    return blocks.transpose(0, 2, 1).reshape(2 * len(x), -1)


def random_field(x, seed):
    rng = np.random.default_rng(seed)
    k = wavevectors(3)
    coef = rng.normal(size=2 * len(k))
    # The exact spatial vector RMS over a periodic unit square is computable.
    w = (1 + np.sum(k * k, axis=1)) ** (-1.5)
    energy = 0.5 * np.sum(
        (coef[: len(k)] ** 2 + coef[len(k) :] ** 2)
        * (2 * np.pi) ** 2
        * np.sum(k * k, axis=1)
        * w
        * w
    )
    return (basis(x, 3) @ coef).reshape(-1, 2) / np.sqrt(energy)


def blind_field(x):
    x = np.asarray(x)
    a, b = 2 * np.pi * x[:, 0], 4 * np.pi * (x[:, 1] - 0.125)
    return np.column_stack(
        (4 * np.pi * np.sin(a) * np.sin(2 * b), -2 * np.pi * np.cos(a) * np.sin(b) ** 2)
    )


def tracks(per_track):
    xx, yy = np.meshgrid(
        np.linspace(0.025, 0.975, per_track), [0.125, 0.375, 0.625, 0.875]
    )
    return np.column_stack((xx.ravel(), yy.ravel())), np.repeat(np.arange(4), per_track)


def target_grid(nx=31, ny=25):
    xx, yy = np.meshgrid(np.linspace(0.025, 0.975, nx), np.linspace(0.135, 0.865, ny))
    return np.column_stack((xx.ravel(), yy.ravel()))


def null_audit(x, q, cutoff=4, rtol=1e-10):
    """Exact-nullspace target vector RMS per unit coefficient norm.

    Numerically exact means singular values <= rtol*s_max. An empty nullspace
    certifies only this finite basis, never the space of smooth fields.
    """
    a, b = basis(x, cutoff), basis(q, cutoff)
    _, s, vh = svd(a, full_matrices=True)
    rank = int(np.sum(s > rtol * s[0])) if len(s) else 0
    z = vh[rank:].T
    if z.shape[1] == 0:
        return dict(
            rank=rank,
            nullity=0,
            radius=0.0,
            residual=0.0,
            coefficients=np.zeros(a.shape[1]),
        )
    _, sb, vbh = svd(b @ z / np.sqrt(len(q)), full_matrices=False)
    c = z @ vbh[0]
    return dict(
        rank=rank,
        nullity=z.shape[1],
        radius=float(sb[0]),
        residual=float(np.max(np.abs(a @ c))),
        coefficients=c,
    )


def noise_audit(x, q, cutoff=4, delta=0.01):
    """Max target RMS under ||c||² + ||A c||²/(n delta²) <= 1.

    This ellipsoid is INSIDE the simultaneous unit coefficient and RMS observation
    budgets, so its radius is a lower bound for that intersection's worst case.
    It is not a probabilistic interval. Uses a symmetric generalized eigenproblem.
    """
    from scipy.linalg import eigh

    if delta <= 0:
        raise ValueError("delta must be positive")
    a, b = basis(x, cutoff), basis(q, cutoff)
    h = b.T @ b / len(q)
    w = np.eye(a.shape[1]) + a.T @ a / (len(x) * delta**2)
    vals, vecs = eigh(h, w, subset_by_index=[len(h) - 1, len(h) - 1])
    c = vecs[:, 0]
    return dict(
        radius=float(np.sqrt(max(vals[0], 0))),
        coefficient_norm=float(np.linalg.norm(c)),
        observation_rms=float(np.linalg.norm(a @ c) / np.sqrt(len(x))),
        coefficients=c,
    )
