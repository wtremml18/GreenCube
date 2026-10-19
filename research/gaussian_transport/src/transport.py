"""Motion and local Gaussian covariance transport; no image renderer."""

import numpy as np


def jacobian(velocity, x, h=1e-5):
    j = np.empty((len(x), 3, 3))
    for axis in range(3):
        delta = np.zeros_like(x)
        delta[:, axis] = h
        j[:, :, axis] = (velocity(x + delta) - velocity(x - delta)) / (2 * h)
    return j


def advect(velocity, centers, covariance, duration=0.5, steps=32):
    """RK4 for dmu/dt=v(mu), dF/dt=Jv(mu)F, Sigma=F Sigma0 F^T.

    Covariance transport is exact for affine flow and first-order local for
    nonlinear flow. Incompressibility preserves det(F) in continuous time;
    ordinary RK4 only approximates that property. No determinant renormalization.
    """
    mu = np.array(centers, dtype=float, copy=True)
    f = np.broadcast_to(np.eye(3), (len(mu), 3, 3)).copy()
    dt = duration / steps

    def rhs(m, g):
        return velocity(m), jacobian(velocity, m) @ g

    for _ in range(steps):
        a, b = rhs(mu, f)
        c, d = rhs(mu + dt * a / 2, f + dt * b / 2)
        e, g = rhs(mu + dt * c / 2, f + dt * d / 2)
        h, j = rhs(mu + dt * e, f + dt * g)
        mu += dt * (a + 2 * c + 2 * e + h) / 6
        f += dt * (b + 2 * d + 2 * g + j) / 6
    sigma = f @ covariance @ f.transpose(0, 2, 1)
    return mu, sigma, f


def flow(x, compressible=False):
    a, b, c = x.T
    v = np.column_stack(
        (
            0.4 * np.sin(c) + 0.3 * np.cos(b),
            0.3 * np.sin(a) + 0.4 * np.cos(c),
            0.4 * np.sin(b) + 0.3 * np.cos(a),
        )
    )
    return v + (0.3 * x if compressible else 0.0)
