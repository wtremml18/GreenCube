"""Exact density-preserving affine motion family (established aperture ambiguity)."""

import numpy as np
from scipy.linalg import expm


def cross_matrix(omega):
    x, y, z = np.asarray(omega)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def generator(scales, omega):
    """A = S Omega S^-1; Sigma=S S^T, A Sigma+Sigma A^T=0."""
    s = np.diag(scales)
    return s @ cross_matrix(omega) @ np.diag(1 / np.asarray(scales))


def sensor_operator(points, scales):
    return np.stack(
        [(points @ generator(scales, w).T).ravel() for w in np.eye(3)], axis=1
    )


def flow_map(scales, omega, time):
    return expm(time * generator(scales, omega))
