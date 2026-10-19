"""Select vector RBF blocks using initial-geometry image derivatives.

No reference volumes are opened. Images are training views only. A one-step
linearization is used for selection; the full nonlinear upstream optimizer
subsequently fits all selected centers, radii and coefficients independently.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--upstream", type=Path, required=True)
p.add_argument("--initial", type=Path, required=True)
p.add_argument("--centers", type=int, default=192)
p.add_argument("--budgets", type=int, nargs="+", default=[32, 64, 128])
p.add_argument("--cache", type=Path, required=True)
p.add_argument("--check-only", action="store_true")
p.add_argument(
    "--stability-sweep",
    action="store_true",
    help="Diagnostic: larger steps and float64 probe accumulation; preserves original check",
)
p.add_argument("--derivative", choices=["autodiff", "finite"], default="autodiff")
a = p.parse_args()
a.upstream, a.initial, a.cache = (
    a.upstream.resolve(),
    a.initial.resolve(),
    a.cache.resolve(),
)
sys.path[:0] = [
    str(ROOT / "src"),
    str(a.upstream),
    str(ROOT.parent / "gaussian_observability/src"),
]
os.chdir(a.upstream)
from argparse import ArgumentParser

import mmcv
import torch
import torch.nn.functional as F
from arguments import PipelineParams
from gaussian_renderer import render
from PIL import Image
from scene.gaussian_loader import load_gaussian_model
from scipy.ndimage import map_coordinates
from selection import block_ols, farthest_order, wendland_blocks
from velocity_common.coordinate_transform import CoordinateTransform
from velocity_common.utils import config_parser, get_background_color
from velocity_training.wrappers import GaussianOverrideWrapper
from visibility import project_points, world_to_voxel

torch.set_default_dtype(torch.float32)
torch.set_default_device("cuda")
args = config_parser().parse_args([])
cfg = mmcv.Config.fromfile(str(ROOT / "configs/selection.py"))
for k, v in cfg.items():
    setattr(args, k, v)
    if k in [
        "ModelParams",
        "OptimizationParams",
        "PipelineParams",
        "ModelHiddenParams",
    ]:
        for kk, vv in v.items():
            setattr(args, kk, vv)
args.eval_frame_limit = None
args.eval = True
args.test_mode = True
args.white_bkgd = np.ones(3, dtype=np.float32)
gaussians, scene = load_gaussian_model(str(a.initial), args.datadir, -1, base_args=args)
pipe = PipelineParams(ArgumentParser()).extract(args)
background = get_background_color(args, "cuda")
meta = json.loads((a.upstream / args.datadir / "info.json").read_text())
vt = np.float32(meta["voxel_matrix"])[:, [2, 1, 0, 3]]
trans = CoordinateTransform(
    vt,
    np.broadcast_to(meta["voxel_scale"], [3]).astype(np.float32).copy()
    * args.scene_scale,
    "cuda",
)
length = torch.tensor([100.0, 150.0, 100.0], device="cuda")
world = gaussians.get_xyz.detach()
xyz = (trans.world2smoke(world) * length).detach()
# Independent coordinate utility agrees with the actual upstream transform.
assert np.allclose(
    xyz.cpu().numpy(), world_to_voxel(world.cpu().numpy(), meta), atol=2e-4
)
cameras = scene.getTrainCameras()
groups = {f: [] for f in range(4)}
for cam in cameras:
    f = round(float(cam.time) * 29)
    if f in groups:
        groups[f].append(cam)
assert all(len(v) == 4 for v in groups.values()), {k: len(v) for k, v in groups.items()}
for f in range(1, 4):
    assert all(
        torch.allclose(c.camera_center, b.camera_center)
        for c, b in zip(groups[f], groups[0])
    )
# Support comes only from frame-zero training images, as in the prior experiment.
w = world.cpu().numpy()
loc = xyz.cpu().numpy()
valid = np.all((loc > [5, 20, 5]) & (loc < [95, 145, 95]), axis=1)
cam_data = json.loads((a.upstream / args.datadir / "transforms_train.json").read_text())
for cam in cam_data["frames"]:
    if cam["time"] != 0:
        continue
    pixels = (
        np.asarray(Image.open(a.upstream / args.datadir / (cam["file_path"] + ".png")))
        .astype(float)[..., :3]
        .mean(-1)
    )
    h, width = pixels.shape
    uv, depth = project_points(
        w,
        cam["transform_matrix"],
        width,
        h,
        cam.get("camera_angle_x", cam_data["camera_angle_x"]),
    )
    intensity = map_coordinates(
        pixels, np.array([uv[:, 1], uv[:, 0]]), order=1, mode="constant", cval=0
    )
    valid &= (depth > 0) & (intensity > 0.01 * pixels.max())
pool = loc[valid]
centers0 = pool[farthest_order(pool, a.centers)]
scales = np.array([30.0, 60.0, 120.0])
centers = np.repeat(centers0, 3, axis=0)
radii = np.tile(scales, len(centers0))
print(
    "Dictionary:",
    len(centers),
    "blocks from",
    len(pool),
    "image-supported points",
    flush=True,
)


def feature(img):
    return F.avg_pool2d(img.mean(0, keepdim=True)[None], 8)[0, 0].flatten()


@torch.no_grad()
def rendered(position):
    ww = trans.smoke2world(position / length)
    wrapped = GaussianOverrideWrapper(gaussians, ww)
    return torch.cat(
        [
            feature(render(cam, wrapped, pipe, background, stage="coarse")["render"])
            for cam in groups[0]
        ]
    )


if a.stability_sweep and not a.check_only:
    raise ValueError("--stability-sweep requires --check-only")
if a.check_only:
    from velocity_common.dfrbf import TiDFRBF
    from velocity_common.rbf import WendlandC4

    rng = np.random.default_rng(191)
    ii = np.array([8, 55, 120, 250, 500])
    model = TiDFRBF(WendlandC4(), centers[ii], radii[ii], device="cuda")
    weight = rng.normal(size=(len(ii), 3)).astype(np.float32)
    model.weights.data.copy_(torch.tensor(weight, device="cuda"))
    with torch.no_grad():
        actual = model(xyz).cpu().numpy()
    expected = np.einsum(
        "ncij,cj->ni", wendland_blocks(loc, centers[ii], radii[ii]), weight
    )
    kernel_error = float(np.max(np.abs(expected - actual)))
    parameter_checks = []
    q = xyz[::97].detach()
    probe_vel = torch.tensor(
        rng.normal(size=(len(q), 3)), dtype=torch.float32, device="cuda"
    )
    loss = (model(q) * probe_vel).sum()
    loss.backward()
    for name, index, step in [
        ("centers", (1, 0), 0.01),
        ("centers", (3, 1), 0.01),
        ("radii", (2,), 0.01),
        ("weights", (2, 1), 0.001),
    ]:
        parameter = getattr(model, name)
        derivative = float(parameter.grad[index])
        original = float(parameter[index])
        values = []
        for sign in [-1, 1]:
            with torch.no_grad():
                parameter[index] = original + sign * step
                values.append(float((model(q) * probe_vel).sum()))
        with torch.no_grad():
            parameter[index] = original
        numeric = (values[1] - values[0]) / (2 * step)
        parameter_checks.append(
            {
                "parameter": name,
                "index": index,
                "analytic": derivative,
                "numeric": numeric,
                "relative_error": abs(numeric - derivative)
                / max(abs(derivative), 1e-3),
            }
        )
    checks = []
    for j in [13, 77, 300]:
        direction = torch.tensor(
            wendland_blocks(loc, centers[j : j + 1], radii[j : j + 1])[:, 0, :, 1],
            dtype=torch.float32,
            device="cuda",
        )
        position = xyz.clone().requires_grad_(True)
        wrapped = GaussianOverrideWrapper(
            gaussians, trans.smoke2world(position / length)
        )
        camera = groups[0][1]
        output = feature(
            render(camera, wrapped, pipe, background, stage="coarse")["render"]
        )
        probe = torch.tensor(
            rng.normal(size=len(output)), dtype=torch.float32, device="cuda"
        )

        def probed(value, probe_vector=probe):
            if a.stability_sweep:
                return (value.double() * probe_vector.double()).sum()
            return (value * probe_vector).sum()

        gradient = torch.autograd.grad(probed(output), position)[0]
        analytic = float((gradient * direction).sum())
        numeric = []
        epsilons = (
            [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
            if a.stability_sweep
            else [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
        )
        for epsilon in epsilons:
            values = []
            for sign in [-1, 1]:
                with torch.no_grad():
                    wrapper = GaussianOverrideWrapper(
                        gaussians,
                        trans.smoke2world((xyz + sign * epsilon * direction) / length),
                    )
                    values.append(
                        float(
                            probed(
                                feature(
                                    render(
                                        camera,
                                        wrapper,
                                        pipe,
                                        background,
                                        stage="coarse",
                                    )["render"]
                                )
                            )
                        )
                    )
            numeric.append((values[1] - values[0]) / (2 * epsilon))
        checks.append(
            {
                "candidate": j,
                "analytic": analytic,
                "epsilons": epsilons,
                "numeric": numeric,
                "relative_error": abs(numeric[2 if a.stability_sweep else 0] - analytic)
                / max(abs(analytic), 1e-3),
                "relative_errors": [
                    abs(v - analytic) / max(abs(analytic), 1e-3) for v in numeric
                ],
            }
        )
    report = {
        "kernel_max_absolute_error": kernel_error,
        "parameter_derivatives": parameter_checks,
        "renderer_derivatives": checks,
        "criterion_epsilon": 0.05 if a.stability_sweep else 0.01,
        "probe_accumulation": "float64" if a.stability_sweep else "float32",
        "passed": kernel_error < 1e-5
        and max(x["relative_error"] for x in checks + parameter_checks) < 0.05,
    }
    (
        ROOT
        / (
            "results/renderer_stability_sweep.json"
            if a.stability_sweep
            else "results/derivative_check.json"
        )
    ).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert report["passed"]
    raise SystemExit(0)

a.cache.parent.mkdir(parents=True, exist_ok=True)
start = time.perf_counter()
if a.cache.exists():
    z = np.load(a.cache)
    assert np.allclose(z["centers"], centers) and np.allclose(z["radii"], radii)
    assert str(z.get("derivative", "finite")) == a.derivative
    jac = z["jacobian"]
    timing = float(z["build_seconds"])
elif a.derivative == "autodiff":
    blocks = wendland_blocks(loc, centers, radii)
    basis_matrix = torch.tensor(
        blocks.transpose(0, 2, 1, 3).reshape(len(loc) * 3, -1),
        dtype=torch.float32,
        device="cuda",
    )
    del blocks
    matrices = []
    adjoints = []
    rng = np.random.default_rng(198)
    for camera_id, camera in enumerate(groups[0]):
        position = xyz.clone().requires_grad_(True)
        wrapped = GaussianOverrideWrapper(
            gaussians, trans.smoke2world(position / length)
        )
        output = feature(
            render(camera, wrapped, pipe, background, stage="coarse")["render"]
        )
        gradient_rows = []
        response = []
        for pixel in range(len(output)):
            gradient = torch.autograd.grad(output[pixel], position, retain_graph=True)[
                0
            ]
            gradient_rows.append(gradient.flatten())
            if len(gradient_rows) == 32 or pixel == len(output) - 1:
                response.append((torch.stack(gradient_rows) @ basis_matrix).detach())
                gradient_rows = []
        matrix = torch.cat(response)
        probe = torch.tensor(
            rng.normal(size=len(output)), dtype=torch.float32, device="cuda"
        )
        direct = (
            torch.autograd.grad((output * probe).sum(), position)[0].flatten()
            @ basis_matrix
        )
        assembled = probe @ matrix
        rel = float(
            torch.linalg.vector_norm(direct - assembled)
            / torch.linalg.vector_norm(direct)
        )
        adjoints.append({"camera": camera_id, "relative_error": rel})
        matrices.append(matrix.cpu().numpy().reshape(len(output), len(centers), 3))
        print(
            "Autodiff camera",
            camera_id,
            "adjoint error",
            rel,
            "elapsed",
            time.perf_counter() - start,
            flush=True,
        )
        del output, matrix, response, position, wrapped
    jac = np.concatenate(matrices)
    timing = time.perf_counter() - start
    report = {
        "passed": max(v["relative_error"] for v in adjoints) < 1e-4,
        "adjoint_checks": adjoints,
        "scope": "Assembly agrees with upstream backward operator; does not resolve finite-difference nonconvergence",
    }
    (ROOT / "results/autodiff_assembly_check.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    assert report["passed"]
    np.savez_compressed(
        a.cache,
        jacobian=jac,
        centers=centers,
        radii=radii,
        build_seconds=timing,
        derivative=a.derivative,
    )
else:
    jac = []
    eps = 0.01
    for j, (center, radius) in enumerate(zip(centers, radii)):
        block = wendland_blocks(loc, center[None], np.array([radius]))[:, 0]
        columns = []
        for k in range(3):
            d = torch.tensor(block[:, :, k], dtype=torch.float32, device="cuda")
            columns.append(
                ((rendered(xyz + eps * d) - rendered(xyz - eps * d)) / (2 * eps))
                .cpu()
                .numpy()
            )
        jac.append(np.stack(columns, axis=-1))
        if j % 48 == 0:
            print(
                "Rendered derivative blocks",
                j,
                "elapsed",
                round(time.perf_counter() - start, 1),
                flush=True,
            )
    jac = np.stack(jac, axis=1)
    timing = time.perf_counter() - start
    a.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.cache,
        jacobian=jac,
        centers=centers,
        radii=radii,
        build_seconds=timing,
        derivative=a.derivative,
    )
np.savez_compressed(
    ROOT
    / (
        "results/autodiff_dictionary.npz"
        if a.derivative == "autodiff"
        else "results/linearized_dictionary.npz"
    ),
    jacobian=jac,
    centers=centers,
    radii=radii,
    build_seconds=timing,
)
# Three one-step temporal differences, using the same initial-geometry Jacobian.
observed = []
for f in range(4):
    observed.append(
        torch.cat([feature(c.original_image.cuda()) for c in groups[f]]).cpu().numpy()
    )
target = np.stack([observed[f + 1] - observed[f] for f in range(3)], axis=1)
image_a = jac / np.sqrt(len(jac))
image_y = target / np.sqrt(len(jac))
sensors = json.loads(
    (
        ROOT.parent / "gaussian_observability/results/image_sensors_visual16.json"
    ).read_text()
)
sx = np.array(sensors["0"]["xyz"])
sa = wendland_blocks(sx, centers, radii).transpose(0, 2, 1, 3).reshape(
    48, len(centers), 3
) / np.sqrt(48)
sy = np.stack(
    [np.array(sensors[str(f)]["velocity"]).ravel() for f in range(3)], axis=1
) / np.sqrt(48)
output = ROOT / "results"
output.mkdir(exist_ok=True)
orders = {}
for method, mat, y in [
    ("image_adols" if a.derivative == "autodiff" else "image_ols", image_a, image_y),
    (
        "joint_adols" if a.derivative == "autodiff" else "joint_ols",
        np.concatenate((image_a, sa)),
        np.concatenate((image_y, sy)),
    ),
]:
    then = time.perf_counter()
    order, trace = block_ols(mat, y, max(a.budgets))
    orders[method] = order
    (output / f"{method}_trace.json").write_text(
        json.dumps(
            {"steps": trace, "selection_seconds": time.perf_counter() - then}, indent=2
        )
        + "\n"
    )
# Exact same support and scale dictionary, without residual-based selection.
orders["coverage"] = [i * 3 + (i % 3) for i in range(max(a.budgets))]
orders["random"] = (
    np.random.default_rng(618)
    .choice(len(centers), max(a.budgets), replace=False)
    .tolist()
)
# Matched-count domain coverage, starting from the actual upstream Poisson set.
from velocity_common.kernels import generate_kernels

pcenters, _ = generate_kernels(np.array([100, 150, 100]), 2500)
poisson = farthest_order(pcenters, max(a.budgets))
for method, order in orders.items():
    for budget in a.budgets:
        ii = np.array(order[:budget])
        (output / f"basis_{method}_{budget}.json").write_text(
            json.dumps(
                {
                    "method": method,
                    "budget": budget,
                    "centers": centers[ii].tolist(),
                    "radii": radii[ii].tolist(),
                    "candidate_indices": ii.tolist(),
                },
                indent=2,
            )
            + "\n"
        )
for budget in a.budgets:
    radius = (100 * 150 * 100 / budget / np.pi * 3 / 4) ** (1 / 3) * 6
    (output / f"basis_poisson_{budget}.json").write_text(
        json.dumps(
            {
                "method": "poisson",
                "budget": budget,
                "centers": pcenters[poisson[:budget]].tolist(),
                "radii": [radius] * budget,
            },
            indent=2,
        )
        + "\n"
    )
np.savez_compressed(
    output / "selection_observations.npz",
    centers=centers,
    radii=radii,
    image_target=target,
    sensor_xyz=sx,
)
(
    output
    / (
        "selection_protocol_autodiff.json"
        if a.derivative == "autodiff"
        else "selection_protocol.json"
    )
).write_text(
    json.dumps(
        {
            "dictionary_blocks": len(centers),
            "candidate_centers": len(centers0),
            "scales": scales.tolist(),
            "image_rows": len(jac),
            "sensor_rows": 48,
            "temporal_intervals": 3,
            "finite_difference_epsilon": 0.01 if a.derivative == "finite" else None,
            "derivative": a.derivative,
            "pool": 8,
            "jacobian_seconds": timing,
            "gt_volumes_used": False,
            "heldout_camera_used": False,
            "selection": "Block OLS on local frame-zero rendering Jacobian; independent interval coefficients",
        },
        indent=2,
    )
    + "\n"
)
print("Saved all basis choices", flush=True)
