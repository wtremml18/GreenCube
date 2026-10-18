"""Potential-GP models with an image-only shear coordinate map.

Coordinates in km. Electric field in mV/m or rotated rocket flow in m/s.
K = T(x) [-d_r d_r^T phi] T(z)^T where q=(x,y-f(x)), T=Dq^T.
A rotated gradient gives a divergence-free flow under a constant-B planar model.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, block_diag


def coordinate_map(x, polynomial, flow=False):
    p = np.polynomial.Polynomial(polynomial)
    q = np.array(x, dtype=float, copy=True)
    q[:, 1] -= p(x[:, 0])
    t = np.broadcast_to(np.eye(2), (len(x), 2, 2)).copy()
    t[:, 0, 1] = -p.deriv()(x[:, 0])
    if flow:
        t = np.einsum('ab,nbc->nac', np.array([[0., 1.], [-1., 0.]]), t)
    return q, t


def covariance(x, z, length, ratio=1., polynomial=(0.,), independent=False, flow=False):
    q, t = coordinate_map(x, polynomial, flow)
    r, u = coordinate_map(z, polynomial, flow)
    ell = np.array([length * ratio, length])
    delta = q[:, None, :] - r[None, :, :]
    phi = np.exp(-0.5 * np.sum((delta / ell) ** 2, axis=2))
    if independent:
        blocks = phi[:, None, :, None] * np.eye(2)[None, :, None, :]
    else:
        d = delta / ell ** 2
        blocks = (np.diag(1 / ell ** 2)[None, None, :, :] - d[:, :, :, None] * d[:, :, None, :]) * (phi * length ** 2)[:, :, None, None]
        blocks = np.einsum('nai,nmij,mbj->namb', t, blocks, u)
    return blocks.reshape(2 * len(x), 2 * len(z))


class PotentialGP:
    def __init__(self, x, y, noise, length, amplitude, floor, ratio=1., polynomial=(0.,), independent=False, flow=False):
        self.x = np.array(x)
        self.kw = dict(length=length, ratio=ratio, polynomial=polynomial, independent=independent, flow=flow)
        self.amplitude = amplitude
        self.floor = floor
        k = amplitude ** 2 * covariance(x, x, **self.kw)
        k += block_diag(*noise) + np.eye(2 * len(x)) * (floor ** 2 + 1e-8)
        self.factor = cho_factor(k, lower=True)
        h = np.tile(np.eye(2), (len(x), 1))
        self.kh = cho_solve(self.factor, h)
        self.normal_inv = np.linalg.inv(h.T @ self.kh)
        ky = cho_solve(self.factor, y.ravel())
        self.mean = self.normal_inv @ (h.T @ ky)
        self.alpha = ky - self.kh @ self.mean

    def predict(self, q, uncertainty=False):
        k = self.amplitude ** 2 * covariance(q, self.x, **self.kw)
        h = np.tile(np.eye(2), (len(q), 1))
        mean = (k @ self.alpha + h @ self.mean).reshape(-1, 2)
        if not uncertainty:
            return mean
        # Universal kriging includes uncertainty from estimating the constant mean.
        solved = cho_solve(self.factor, k.T)
        residual = h - k @ self.kh
        full = self.amplitude ** 2 * covariance(q, q, **self.kw) - k @ solved + residual @ self.normal_inv @ residual.T
        blocks = np.stack([full[2*i:2*i+2, 2*i:2*i+2] for i in range(len(q))])
        return mean, (blocks + blocks.transpose(0, 2, 1)) / 2


def model_options(name, polynomial, flow=False):
    p = np.array(polynomial)
    options = dict(flow=flow)
    if name == 'independent':
        options['independent'] = True
    if name in ('straight_anisotropic', 'image_aligned', 'image_linear', 'image_reversed'):
        options['ratio'] = 4.
    if name == 'image_aligned':
        options['polynomial'] = p
    if name == 'image_linear':
        options['polynomial'] = [p[0], p[1]]
    if name == 'image_reversed':
        options['polynomial'] = -p
    return options


def contiguous_splits(ids, count=4, guard=1):
    """Original row IDs define adjacency even after filtering or an outer holdout."""
    ids = np.asarray(ids)
    for held in np.array_split(np.arange(len(ids)), count):
        if not len(held):
            continue
        gap = np.min(abs(ids[:, None] - ids[held][None, :]), axis=1)
        train = np.flatnonzero(gap > guard)
        if len(train) >= 3:
            yield train, held


def fit_tuned(x, y, noise, ids, name, polynomial, flow=False, output_scale=None):
    opts = model_options(name, polynomial, flow)
    lengths = [10., 30., 90.] if flow else [20., 60., 180.]
    amplitudes = [200., 600., 1800.] if flow else [10., 30., 100.]
    floors = [0., 100.] if flow else [0., 10.]
    splits = list(contiguous_splits(ids, 3, guard=2 if flow else 1))
    if output_scale is None:
        output_scale = np.ones_like(y)
    if len(splits) < 2:
        raise ValueError('Insufficient training groups for nested tuning')
    best = None
    for length in lengths:
        for amplitude in amplitudes:
            for floor in floors:
                loss = 0.; n = 0
                for tr, te in splits:
                    model = PotentialGP(x[tr], y[tr], noise[tr], length, amplitude, floor, **opts)
                    loss += np.sum(((model.predict(x[te]) - y[te]) * output_scale[te]) ** 2)
                    n += len(te)
                row = dict(length=length, amplitude=amplitude, floor=floor, inner_mse=loss/n)
                if best is None or row['inner_mse'] < best['inner_mse']:
                    best = row
    model = PotentialGP(x, y, noise, **{k:best[k] for k in ['length','amplitude','floor']}, **opts)
    return model, best
