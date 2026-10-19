"""Validate saved evaluation counts, finite errors and held-out sample separation."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
expected = {
    "track_observability/results/synthetic.csv": 480,
    "track_observability/results/design.csv": 160,
    "track_observability/results/public.csv": 24,
    "track_observability/results/dipole_robustness.csv": 120,
    "track_observability/results/public_robustness.csv": 120,
    "track_observability/results/geometry.csv": 100,
    "gaussian_transport/results/transport.csv": 120,
    "gaussian_transport/results/fluid.csv": 180,
    "gaussian_transport/results/fluid_support_matched.csv": 180,
}
for name, count in expected.items():
    df = pd.read_csv(ROOT / name)
    assert len(df) == count, (name, len(df), count)
    assert np.isfinite(df.select_dtypes(include="number").to_numpy()).all(), name
for suffix in ["", "_support_matched"]:
    rows = json.loads(
        (
            ROOT / "gaussian_transport/results" / f"fluid_samples{suffix}.json"
        ).read_text()
    )
    assert len(rows) == 60
    reference = {}
    for row in rows:
        train = set(map(tuple, row["train_zyx"]))
        test = set(map(tuple, row["test_zyx"]))
        assert len(train) == 96 and len(test) == 512
        assert train.isdisjoint(test)
        key = (row["frame"], row["seed"])
        if key in reference:
            assert reference[key] == test
        else:
            reference[key] = test
matched = pd.read_csv(ROOT / "gaussian_transport/results/fluid_support_matched.csv")
assert np.all(matched.training_smoke_fraction == 1.0)
splits = json.loads(
    (ROOT / "track_observability/results/public_splits.json").read_text()
)
for s in splits.values():
    assert len(set(s["original_rows"])) == len(s["original_rows"])
    assert set(s["train"]).isdisjoint(s["test"])
    for inner in s["inner"]:
        assert set(inner["train"]).isdisjoint(inner["test"])
        assert set(inner["train"] + inner["test"]) == set(range(len(s["train"])))
print(
    f"Validated {sum(expected.values())} evaluation rows, 120 fluid sample designs, and nested public splits."
)
