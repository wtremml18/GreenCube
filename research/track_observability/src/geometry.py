"""Piola-transformed kernels for electrostatic weighted flux in a curved plane.

Known construction, not a new kernel claim. Map X(s,n)=(s,n+a sin(2pi s));
det(DX)=1. If div_q u=0 then v(X(q))=DX(q)u(q)/B(X(q)) has div_x(Bv)=0.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .kernels import kernel


def geometry(x, amplitude=0.0, beta=0.0):
    q = np.array(x, dtype=float, copy=True)
    q[:, 1] -= amplitude * np.sin(2 * np.pi * q[:, 0])
    t = np.broadcast_to(np.eye(2), (len(x), 2, 2)).copy()
    t[:, 1, 0] = 2 * np.pi * amplitude * np.cos(2 * np.pi * q[:, 0])
    t /= np.exp(beta * x[:, 0, None, None])
    return q, t


def push(q, amplitude):
    x = np.array(q, copy=True)
    x[:, 1] += amplitude * np.sin(2 * np.pi * q[:, 0])
    return x


def warped_kernel(x, z, length, amplitude, beta):
    q, t = geometry(x, amplitude, beta)
    r, u = geometry(z, amplitude, beta)
    blocks = kernel(q, r, length).reshape(len(x), 2, len(z), 2)
    return np.einsum("nai,nimj,mbj->namb", t, blocks, u).reshape(2 * len(x), 2 * len(z))


class WarpedModel:
    def __init__(self, x, y, length, ridge, amplitude, beta):
        self.x = x
        self.length = length
        self.amplitude = amplitude
        self.beta = beta
        k = warped_kernel(x, x, length, amplitude, beta)
        k.flat[:: len(k) + 1] += ridge
        factor = cho_factor(k, lower=True)
        # Joint generalized least-squares constant background in reference flux.
        h = geometry(x, amplitude, beta)[1].reshape(-1, 2)
        kh = cho_solve(factor, h)
        ky = cho_solve(factor, y.ravel())
        self.background = np.linalg.solve(h.T @ kh, h.T @ ky)
        self.alpha = ky - kh @ self.background

    def predict(self, q):
        h = geometry(q, self.amplitude, self.beta)[1]
        k = warped_kernel(q, self.x, self.length, self.amplitude, self.beta)
        return (k @ self.alpha).reshape(-1, 2) + h @ self.background


def tune_warped(x, y, splits, amplitude, beta):
    scores = []
    for length in [0.08, 0.16, 0.32, 0.64]:
        for ridge in [1e-4, 0.01, 0.1]:
            sse = 0.0
            n = 0
            for tr, te in splits:
                m = WarpedModel(x[tr], y[tr], length, ridge, amplitude, beta)
                sse += np.sum((m.predict(x[te]) - y[te]) ** 2)
                n += len(te)
            scores.append(dict(length=length, ridge=ridge, cv_mse=sse / n))
    b = min(scores, key=lambda row: row["cv_mse"])
    return WarpedModel(x, y, b["length"], b["ridge"], amplitude, beta), b
