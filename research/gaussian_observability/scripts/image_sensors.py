"""Place instruments from image-fitted Gaussians; sample public simulated velocity.

Location selection uses only the initial image reconstruction, never velocity GT.
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from design import farthest
from visibility import world_to_voxel, project_points

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--data", type=Path, required=True)
p.add_argument("--info", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--count", type=int, default=16)
p.add_argument(
    "--selection", choices=["opacity", "brightness", "silhouettes"], default="opacity"
)
p.add_argument("--frames", type=int, default=3)
p.add_argument("--placement", choices=["farthest", "random"], default="farthest")
p.add_argument("--placement-seed", type=int, default=413)
a = p.parse_args()
# Only our locally generated checkpoint is loaded.
state, _ = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
world = state[1].detach().numpy()
opacity = torch.sigmoid(state[8]).detach().numpy().ravel()
scale = np.exp(state[6].detach().numpy())
meta = json.loads(a.info.read_text())
xyz = world_to_voxel(world, meta)
valid = np.all((xyz > [5, 20, 5]) & (xyz < [95, 145, 95]), axis=1)
mass = opacity * np.prod(scale, axis=1)
if a.selection == "brightness":
    color = np.maximum(
        0, 0.5 + 0.28209479177387814 * state[4].detach().numpy()[:, 0, :]
    ).mean(axis=1)
    mass = mass * color
    # Retain the strongest emitters accounting for 90% of visible emission proxy.
    order = np.argsort(mass[valid])[::-1]
    weights = mass[valid][order]
    stop = np.searchsorted(np.cumsum(weights), 0.9 * weights.sum())
    cut = weights[min(stop, len(weights) - 1)]
else:
    cut = np.quantile(mass[valid], 0.5)
if a.selection == "silhouettes":
    from PIL import Image

    camera_data = json.loads((a.info.parent / "transforms_train.json").read_text())
    silhouette = np.ones(len(world), dtype=bool)
    for camera in camera_data["frames"]:
        if camera["time"] != 0:
            continue
        # Match upstream _process_single_frame's actual convention (c2w/OpenGL).
        pixels = (
            np.asarray(Image.open(a.info.parent / (camera["file_path"] + ".png")))
            .astype(float)[..., :3]
            .mean(axis=-1)
        )
        h, w = pixels.shape
        uv, depth = project_points(
            world,
            camera["transform_matrix"],
            w,
            h,
            camera.get("camera_angle_x", camera_data["camera_angle_x"]),
        )
        u, v = uv.T
        intensity = map_coordinates(
            pixels, np.array([v, u]), order=1, mode="constant", cval=0
        )
        silhouette &= (depth > 0) & (intensity > 0.01 * pixels.max())
    candidates = xyz[valid & silhouette]
else:
    candidates = xyz[valid & (mass >= cut)]
if len(candidates) < a.count:
    raise ValueError(f"Insufficient image-supported candidates: {len(candidates)}")
selected = []
for _ in range(a.count):
    selected.append(farthest(candidates, selected))
if a.placement == "random":
    selected = np.random.default_rng(a.placement_seed).choice(
        len(candidates), a.count, replace=False
    )
locations = candidates[selected]
rng = np.random.default_rng(20261018)
result = {}
for frame in range(a.frames):
    vel = np.load(a.data / f"velocity_{frame:04d}.npz")["arr_0"][:100, :150, :100]
    readings = np.stack(
        [
            map_coordinates(vel[..., j], locations[:, ::-1].T, order=1, mode="nearest")
            for j in range(3)
        ],
        axis=1,
    )
    noisy = readings + rng.normal(0, 0.05, readings.shape)
    result[str(frame)] = dict(xyz=locations.tolist(), velocity=noisy.tolist())
result["metadata"] = dict(
    count=a.count,
    source=f"initial image-fitted Gaussian centers; {a.placement} selection",
    selection=a.selection,
    placement=a.placement,
    placement_seed=a.placement_seed,
    cutoff=float(cut),
    noise_sigma=0.05,
    checkpoint_name=a.checkpoint.name,
    candidate_count=len(candidates),
    coordinate_system="native voxel xyz",
    velocity_units="native released ScalarSyn velocity",
    effective_frame_dt=1.0,
)
a.out.write_text(json.dumps(result, indent=2) + "\n")
print(result["metadata"])
