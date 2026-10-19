"""Check sparse-instrument gradient through the upstream CUDA/Taichi model."""

import argparse, sys, json
from pathlib import Path
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--upstream", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.upstream.resolve()))
from velocity_common.dfrbf import TiDFRBF
from velocity_common.rbf import WendlandC4

rng = np.random.default_rng(22)
m = TiDFRBF(WendlandC4(), rng.uniform(-1, 1, (12, 3)), 2.0, device="cuda")
with torch.no_grad():
    m.weights.copy_(
        torch.tensor(rng.normal(0, 0.01, (12, 3)), dtype=torch.float32, device="cuda")
    )
x = torch.tensor(rng.uniform(-0.5, 0.5, (8, 3)), dtype=torch.float32, device="cuda")
y = torch.tensor(rng.normal(0, 0.1, (8, 3)), dtype=torch.float32, device="cuda")
loss = ((m(x) - y) ** 2).mean()
loss.backward()
g = m.weights.grad.clone()
indices = torch.topk(g.abs().flatten(), 5).indices.cpu().numpy()
rows = []
for flat in indices:
    i, j = divmod(int(flat), 3)
    w = float(m.weights[i, j])
    eps = 1e-4
    with torch.no_grad():
        m.weights[i, j] = w + eps
        plus = float(((m(x) - y) ** 2).mean())
        m.weights[i, j] = w - eps
        minus = float(((m(x) - y) ** 2).mean())
        m.weights[i, j] = w
    numeric = (plus - minus) / (2 * eps)
    analytic = float(g[i, j])
    relative = abs(numeric - analytic) / max(abs(numeric), 1e-8)
    assert relative < 0.01, (numeric, analytic)
    rows.append(
        dict(index=[i, j], numeric=numeric, analytic=analytic, relative_error=relative)
    )
a.out.write_text(json.dumps(dict(passed=True, derivatives=rows), indent=2) + "\n")
print(rows)
