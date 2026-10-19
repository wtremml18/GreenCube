"""Gaussian linear conditioning and A-optimal vector sensor design.

These are established experimental-design formulas, not new kernel mathematics.
All selection functions accept coordinates/covariances only, never truth values.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def condition(mean, covariance, indices, value, noise_variance):
    indices = np.asarray(indices)
    cross = covariance[:, indices]
    innovation = covariance[np.ix_(indices, indices)] + noise_variance * np.eye(
        len(indices)
    )
    gain = cho_solve(cho_factor(innovation, lower=True), cross.T).T
    new_mean = mean + gain @ (value - mean[indices])
    new_cov = covariance - gain @ cross.T
    return new_mean, (new_cov + new_cov.T) / 2


def scores(covariance, candidates, targets, noise_variance, criterion="integrated"):
    blocks = np.asarray(candidates)[:, None] * 3 + np.arange(3)
    local = covariance[blocks[:, :, None], blocks[:, None, :]]
    if criterion == "variance":
        return np.trace(local, axis1=1, axis2=2)
    inv = np.linalg.inv(local + noise_variance * np.eye(3))
    cross = covariance[np.asarray(targets)[:, None], blocks.reshape(-1)].reshape(
        len(targets), -1, 3
    )
    return np.einsum("tci,cij,tcj->c", cross, inv, cross, optimize=True)


def farthest(points, selected):
    if not selected:
        # Reproducible central start, without any field values.
        return int(np.argmin(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))
    d2 = np.min(np.sum((points[:, None] - points[selected]) ** 2, axis=-1), axis=1)
    d2[selected] = -np.inf
    return int(np.argmax(d2))
