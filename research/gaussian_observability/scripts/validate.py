"""Audit saved scientific evidence, common holdouts and experiment completeness."""

from pathlib import Path
import gzip, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for tag in ["primary", "short", "long", "discrepancy", "independent"]:
    d = pd.read_csv(ROOT / "results" / f"design_{tag}.csv")
    assert len(d) == 2400, (tag, len(d))
    assert np.isfinite(d.select_dtypes("number").to_numpy()).all()
    assert set(d.frame) == {1, 10, 20, 28} and set(d.seed) == set(range(10))
    config = json.loads((ROOT / "results" / f"config_{tag}.json").read_text())
    assert config["effective_frame_dt"] == 1
    layouts = json.loads(
        gzip.decompress((ROOT / "results" / f"selections_{tag}.json.gz").read_bytes())
    )
    assert len(layouts) == 480
    for r in layouts:
        obs = set(map(tuple, r["density_zyx"]))
        test = set(map(tuple, r["test_zyx"]))
        sensor = set(map(tuple, r["selected_zyx"]))
        assert len(obs) == 256 and len(test) == 256 and len(sensor) == 16
        assert (
            obs.isdisjoint(test) and obs.isdisjoint(sensor) and test.isdisjoint(sensor)
        )
        assert len(r["selected_zyx"]) == 16
samples = np.load(ROOT / "results/common_samples.npz")
protocol = json.loads((ROOT / "results/common_sample_protocol.json").read_text())
locations = np.array(protocol["all_sensor_xyz"])
for frame in range(9):
    idx = samples[f"indices_frame{frame}"]
    assert idx.shape == (4096, 3)
    assert len(np.unique(idx, axis=0)) == 4096
    assert np.min(np.sum((idx[:, None, ::-1] - locations) ** 2, axis=-1)) > 4
    truth = samples[f"truth_frame{frame}"]
    assert truth.shape == (4096, 3) and np.isfinite(truth).all()
common = pd.read_csv(ROOT / "results/common_motion.csv")
assert len(common) == 114, len(common)
assert common.run_name.nunique() == 24, common.run_name.nunique()
assert np.isfinite(common.select_dtypes("number").to_numpy()).all()
predictions = np.load(ROOT / "results/common_motion_predictions.npz")
for r in common.itertuples():
    key = f"{r.condition}_s{r.seed}_w{r.weight:g}_frame{r.frame}"
    prediction = predictions[key]
    truth = samples[f"truth_frame{r.frame}"]
    assert prediction.shape == truth.shape and np.isfinite(prediction).all()
    loss = np.sqrt(np.mean(np.sum((prediction - truth) ** 2, axis=1)))
    assert np.isclose(loss, r.vector_rmse, rtol=1e-6, atol=1e-8)
images = list((ROOT / "results").glob("images_*_s*_w*.json"))
assert len(images) == 34, len(images)
for path in images:
    d = json.loads(path.read_text())
    n = len(d["velocity"]) + 1
    assert n in [4, 10]
    assert d["images"]["train"]["num_cameras"] == 4 * n
    assert d["images"]["test"]["num_cameras"] == n
    assert d["images"]["test"]["per_camera"]["frame_indices"] == list(range(n))
    name = path.stem.removeprefix("images_")
    condition, rest = name.split("_s", 1)
    seed, weight = rest.split("_w", 1)
    match = common[
        (common.condition == condition)
        & (common.seed == int(seed))
        & (common.weight == float(weight))
    ]
    assert set(match.frame) == set(range(n - 1))
sensor = pd.read_csv(ROOT / "results/sensor_only.csv")
assert len(sensor) == 66, len(sensor)
assert np.isfinite(sensor.select_dtypes("number").to_numpy()).all()
sensor_predictions = np.load(ROOT / "results/sensor_only_predictions.npz")
for r in sensor.itertuples():
    prediction = sensor_predictions[f"{r.layout}_{r.kind}_frame{r.frame}"]
    truth = samples[f"truth_frame{r.frame}"]
    loss = np.sqrt(np.mean(np.sum((prediction - truth) ** 2, axis=1)))
    assert np.isclose(loss, r.vector_rmse, rtol=1e-6, atol=1e-8)
roi = pd.read_csv(ROOT / "results/image_roi.csv")
assert len(roi) == 148, len(roi)
components = pd.read_csv(ROOT / "results/motion_components.csv")
assert len(components) == 180
assert np.isfinite(components.select_dtypes("number").to_numpy()).all()
assert np.allclose(
    components.normal_mse + components.tangent_vector_mse, components.total_vector_mse
)
gpu = json.loads((ROOT / "results/gpu_gradient_check.json").read_text())
assert gpu["passed"]
assert max(d["relative_error"] for d in gpu["derivatives"]) < 0.01
report = dict(
    passed=True,
    volumetric_metric_rows=12000,
    camera_evaluation_conditions=len(images),
    unique_controlled_camera_fits=int(common.run_name.nunique()),
    common_motion_frame_rows=len(common),
    common_motion_vector_predictions=int(common.n.sum()),
    sensor_only_frame_rows=len(sensor),
    prediction_loss_recomputation_passed=True,
    reserved_camera_frame_scores=len(roi),
    directional_diagnostic_rows=len(components),
    cpu_unit_tests=7,
    gpu_gradient_checks=len(gpu["derivatives"]),
    min_sensor_exclusion_voxels=2,
    independent_simulations=1,
)
(ROOT / "results/verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
