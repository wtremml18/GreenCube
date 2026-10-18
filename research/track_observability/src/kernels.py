"""Normalized vector Gaussian kernels; independent from legacy MATLAB code."""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def kernel(x, z, length, kind="df", mix=0.0):
    """K_df = (Hessian - I Laplacian) phi, normalized to K(0)=I.

    phi=exp(-r²/(2 l²)); supports d>=2. K_cf=-Hessian(phi), also normalized.
    ``mix`` weights the curl-free part, so mix=0 is solenoidal.
    """
    x, z = np.asarray(x), np.asarray(z)
    d = x.shape[1]
    if d < 2 or z.shape[1] != d or length <= 0:
        raise ValueError("Need matching dimension >=2 and positive length")
    r = (x[:, None, :] - z[None, :, :]) / length
    r2 = np.sum(r * r, axis=-1)
    phi = np.exp(-r2 / 2)
    eye = np.eye(d)
    outer = r[:, :, :, None] * r[:, :, None, :]
    if kind == "independent":
        out = phi[:, :, None, None] * eye
    elif kind in ("df", "helmholtz"):
        df = ((d - 1 - r2)[:, :, None, None] * eye + outer) / (d - 1)
        cf = eye - outer
        if kind == "df":
            mix = 0.0
        if not 0 <= mix <= 1:
            raise ValueError("mix must be in [0,1]")
        out = phi[:, :, None, None] * ((1 - mix) * df + mix * cf)
    else:
        raise ValueError(kind)
    return out.transpose(0, 2, 1, 3).reshape(len(x) * d, len(z) * d)


@dataclass
class Model:
    x: np.ndarray
    mean: np.ndarray
    scale: float
    alpha: np.ndarray
    length: float
    ridge: float
    kind: str
    mix: float

    def predict(self, q):
        return (
            kernel(q, self.x, self.length, self.kind, self.mix) @ self.alpha
        ).reshape(len(q), -1) * self.scale + self.mean


def fit(x, y, length, ridge, kind="df", mix=0.0):
    if ridge <= 0 or not np.all(np.isfinite(y)):
        raise ValueError("Positive ridge and finite observations required")
    mean = np.mean(y, axis=0)
    scale = max(float(np.sqrt(np.mean((y - mean) ** 2))), 1e-12)
    k = kernel(x, x, length, kind, mix)
    k.flat[:: len(k) + 1] += ridge
    alpha = cho_solve(cho_factor(k, lower=True), ((y - mean) / scale).ravel())
    return Model(x, mean, scale, alpha, length, ridge, kind, mix)


def folds(n, labels=None, seed=0):
    if labels is None:
        labels = np.empty(n, dtype=int)
        labels[np.random.default_rng(seed).permutation(n)] = np.arange(n) % 4
    out = []
    for label in np.unique(labels):
        test = np.flatnonzero(labels == label)
        train = np.flatnonzero(labels != label)
        if len(train) >= 3 and len(test):
            out.append((train, test))
    if len(out) < 2:
        raise ValueError("Need at least two nonempty validation groups")
    return out


def tune(
    x, y, splits, kind="df", lengths=(0.08, 0.16, 0.32, 0.64), ridges=(1e-4, 0.01, 0.1)
):
    mixes = (0.0, 0.25, 0.5, 0.75, 1.0) if kind == "helmholtz" else (0.0,)
    scores = []
    for mix in mixes:
        for length in lengths:
            for ridge in ridges:
                sse = 0.0
                n = 0
                for train, test in splits:
                    model = fit(x[train], y[train], length, ridge, kind, mix)
                    sse += np.sum((model.predict(x[test]) - y[test]) ** 2)
                    n += len(test)
                scores.append(dict(length=length, ridge=ridge, mix=mix, cv_mse=sse / n))
    best = min(scores, key=lambda v: v["cv_mse"])
    model = fit(x, y, best["length"], best["ridge"], kind, best["mix"])
    return model, best, scores


def divergence(model, q, h=1e-5):
    ans = np.zeros(len(q))
    for d in range(q.shape[1]):
        delta = np.zeros_like(q)
        delta[:, d] = h
        ans += (model.predict(q + delta)[:, d] - model.predict(q - delta)[:, d]) / (
            2 * h
        )
    return ans
