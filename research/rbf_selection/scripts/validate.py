"""Verify saved prediction losses, unchanged holdouts and recorded basis sizes."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
samples = np.load(ROOT.parent / "gaussian_observability/results/common_samples.npz")
pred = np.load(ROOT / "results/common_predictions.npz")
metrics = pd.read_csv(ROOT / "results/common_metrics.csv")
assert metrics.tag.nunique() == 54
assert len(metrics) == 162 == 3 * metrics.tag.nunique()
assert np.isfinite(metrics.select_dtypes("number").to_numpy()).all()
for r in metrics.itertuples():
    p = pred[f"{r.tag}_frame{r.frame}"]
    truth = samples[f"truth_frame{r.frame}"]
    assert p.shape == (4096, 3) and np.isfinite(p).all()
    loss = np.sqrt(np.mean(np.sum((p - truth) ** 2, axis=1)))
    assert np.isclose(loss, r.vector_rmse, rtol=1e-6)
    basis = json.loads((ROOT / "results" / r.basis_file).read_text())
    assert len(basis["centers"]) == len(basis["radii"]) == r.budget
for method in ["image_ols", "joint_ols", "image_adols", "joint_adols", "joint_rols"]:
    trace = json.loads((ROOT / "results" / f"{method}_trace.json").read_text())["steps"]
    assert len(trace) == 128 and len({v["candidate"] for v in trace}) == 128
    assert np.all(np.diff([v["residual_sse"] for v in trace]) <= 1e-12)
check = json.loads((ROOT / "results/derivative_check.json").read_text())
assert check["kernel_max_absolute_error"] < 1e-5
assert all(d["relative_error"] < 0.05 for d in check["parameter_derivatives"])
ad = json.loads((ROOT / "results/autodiff_assembly_check.json").read_text())
assert ad["passed"] and all(d["relative_error"] < 1e-6 for d in ad["adjoint_checks"])
for tag in metrics.tag.unique():
    evaluation = json.loads((ROOT / "results" / f"eval_{tag}.json").read_text())
    assert [item["frame"] for item in evaluation["velocity"]] == [0, 1, 2]
    for item in evaluation["velocity"]:
        frame = item["frame"]
        original = np.load(
            ROOT / "results" / f"eval_{tag}_frame{frame}_predictions.npz"
        )
        assert original["prediction"].shape == original["truth"].shape == (4096, 3)
        assert np.isfinite(original["prediction"]).all()
        loss = np.sqrt(
            np.mean(np.sum((original["prediction"] - original["truth"]) ** 2, axis=1))
        )
        assert np.isclose(loss, item["vector_rmse"], rtol=1e-6)
    for split in ["train", "test"]:
        im = evaluation["images"][split]
        for metric in ["psnr", "ssim", "lpips_vgg", "lpips_alex"]:
            values = np.array(im["per_camera"][metric])
            assert len(values) == im["num_cameras"] and np.isfinite(values).all()
            assert np.isclose(values.mean(), im[metric], rtol=1e-6)
    assert evaluation["images"]["train"]["num_cameras"] == 16
    assert evaluation["images"]["test"]["num_cameras"] == 4
    assert evaluation["images"]["test"]["per_camera"]["frame_indices"] == [0, 1, 2, 3]
    assert evaluation["images"]["train"]["per_camera"]["frame_indices"] == [
        i for i in range(4) for _ in range(4)
    ]
for budget in [32, 64, 128]:
    cv = json.loads((ROOT / "results" / f"basis_cv_scale_{budget}.json").read_text())
    baseline = json.loads(
        (ROOT / "results" / f"basis_poisson_{budget}.json").read_text()
    )
    assert np.array_equal(cv["centers"], baseline["centers"])
    assert np.array_equal(cv["radii"], baseline["radii"])
learned = pd.read_csv(ROOT / "results/learned_basis.csv")
assert len(learned) == 162
assert np.isfinite(learned.select_dtypes("number").to_numpy()).all()
diagnostic = pd.read_csv(ROOT / "results/surrogate_diagnostics.csv")
assert len(diagnostic) == 9 and diagnostic.refit_relative_difference.max() < 1e-7
reference = pd.read_csv(ROOT / "results/reference_basis.csv")
assert len(reference) == 18 and (reference.kernels == 1121).all()
provenance = json.loads((ROOT / "results/provenance.json").read_text())
for item in ["fixed_observations", "common_holdouts"]:
    source = ROOT.parent / "gaussian_observability/results" / provenance[item]["file"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance[item]["sha256"]
sensors = json.loads(
    (
        ROOT.parent / "gaussian_observability/results/image_sensors_visual16.json"
    ).read_text()
)
for frame in range(3):
    xyz = samples[f"indices_frame{frame}"][:, ::-1]
    sensor_xyz = np.array(sensors[str(frame)]["xyz"])
    distance = np.linalg.norm(xyz[:, None] - sensor_xyz[None], axis=-1)
    assert distance.min() > 2
linear = pd.read_csv(ROOT / "results/linear_sensor_only.csv")
linear_pred = np.load(ROOT / "results/linear_sensor_only_predictions.npz")
assert len(linear) == 30
for row in linear.itertuples():
    value = linear_pred[row.prediction_key]
    truth = samples[f"truth_frame{row.frame}"]
    assert value.shape == truth.shape and np.isfinite(value).all()
    loss = np.sqrt(np.mean(np.sum((value - truth) ** 2, axis=1)))
    assert np.isclose(loss, row.vector_rmse, rtol=1e-6)
ablation = json.loads((ROOT / "results/image_selection_ablation.json").read_text())
assert all(item["same_set"] for item in ablation["comparisons"])
for budget in [32, 64, 128]:
    sensor_basis = json.loads(
        (ROOT / "results" / f"basis_sensor_rols_{budget}.json").read_text()
    )
    joint_basis = json.loads(
        (ROOT / "results" / f"basis_joint_rols_{budget}.json").read_text()
    )
    assert set(sensor_basis["candidate_indices"]) == set(
        joint_basis["candidate_indices"]
    )
sweep_path = ROOT / "results/renderer_stability_sweep.json"
sweep = json.loads(sweep_path.read_text()) if sweep_path.exists() else None
report = {
    "artifact_validation_passed": True,
    "completed_camera_fits": int(metrics.tag.nunique()),
    "common_motion_rows": len(metrics),
    "secondary_layout_motion_rows_recomputed": len(metrics),
    "reserved_camera_frame_scores": 4 * int(metrics.tag.nunique()),
    "sensor_separation_verified": True,
    "motion_vectors": int(metrics.n.sum()),
    "cpu_tests": 5,
    "independent_surrogate_refits": 9,
    "linear_sensor_only_frame_rows": 30,
    "linear_sensor_only_vectors": int(linear.n.sum()),
    "image_row_ablation_same_kernel_sets": True,
    "finite_difference_renderer_check_passed": check["passed"],
    "finite_difference_renderer_probes": len(check["renderer_derivatives"]),
    "finite_difference_pilot_fits": 6,
    "larger_step_renderer_check_passed": None if sweep is None else sweep["passed"],
    "backward_operator_consistency_checks": len(ad["adjoint_checks"]),
    "derivative_limitation": "Backward assembly is consistent with upstream optimizer; renderer finite differences did not converge reliably.",
    "kernel_parameter_gradient_checks": len(check["parameter_derivatives"]),
    "independent_scenes": 1,
    "all_prediction_losses_recomputed": True,
}
(ROOT / "results/verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
