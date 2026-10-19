"""Evaluate reserved camera and independent smoke voxels for an upstream run."""

import argparse, json, sys, os
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--upstream", type=Path, required=True)
p.add_argument("--run", type=Path, required=True)
p.add_argument("--config", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--sensors", type=Path, required=True)
p.add_argument("--frames", type=int, default=4)
a = p.parse_args()
a.upstream = a.upstream.resolve()
a.run = a.run.resolve()
a.config = a.config.resolve()
a.out = a.out.resolve()
a.sensors = a.sensors.resolve()
sys.path.insert(0, str(a.upstream))
os.chdir(a.upstream)
import torch, mmcv
from velocity_common.utils import config_parser
from velocity_common.dfrbf import TiDFRBF
from velocity_evaluation.evaluate import evaluate_sliding_window_window

torch.set_default_dtype(torch.float32)
torch.set_default_device("cuda")
args = config_parser().parse_args([])
cfg = mmcv.Config.fromfile(str(a.config))
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
args.model_path = str(a.run)
args.source_path = args.datadir
args.eval = True
args.test_mode = True
args.white_bkgd = np.ones(3, dtype=np.float32)
# Keep the true 30-frame time normalization; evaluate only window [0,4).
args.eval_frame_limit = None
images = evaluate_sliding_window_window(args, str(a.run), 0, a.frames, scale=args.scale)
velocity = []
for frame in range(a.frames - 1):
    rho = np.load(f"gt/scalarsyn/density_{frame:04d}.npz")["arr_0"][:100, :150, :100, 0]
    gt = np.load(f"gt/scalarsyn/velocity_{frame:04d}.npz")["arr_0"][:100, :150, :100]
    support = rho > 0.01 * rho.max()
    support[:5] = support[-5:] = False
    support[:, :20] = support[:, -5:] = False
    support[:, :, :5] = support[:, :, -5:] = False
    pool = np.argwhere(support)
    locations = np.array(json.loads(a.sensors.read_text())[str(frame)]["xyz"])
    far = np.min(np.sum((pool[:, None, ::-1] - locations) ** 2, axis=-1), axis=1) > 4
    pool = pool[far]
    idx = pool[
        np.random.default_rng(7400 + frame).choice(len(pool), 4096, replace=False)
    ]
    x = torch.tensor(idx[:, ::-1].copy(), dtype=torch.float32, device="cuda")
    model = TiDFRBF.load(
        str(a.run / "frame_velocities" / f"frame_{frame:03d}_velocity.pth"),
        device="cuda",
    )
    with torch.no_grad():
        prediction = model(x).cpu().numpy()
    truth = gt[tuple(idx.T)]
    velocity.append(
        dict(
            frame=frame,
            n=len(idx),
            vector_rmse=float(
                np.sqrt(np.mean(np.sum((prediction - truth) ** 2, axis=1)))
            ),
            zero_rmse=float(np.sqrt(np.mean(np.sum(truth**2, axis=1)))),
            mean_speed=float(np.linalg.norm(prediction, axis=1).mean()),
        )
    )
    np.savez_compressed(
        a.out.with_name(a.out.stem + f"_frame{frame}_predictions.npz"),
        indices_zyx=idx,
        truth=truth,
        prediction=prediction,
    )
result = dict(run_name=a.run.name, images=images, velocity=velocity)
a.out.write_text(json.dumps(result, indent=2, default=lambda x: float(x)) + "\n")
print(json.dumps(result, indent=2, default=lambda x: float(x)))
