"""Matched camera/sensor input, fixed epochs, exact selected-kernel count."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--upstream", type=Path, required=True)
p.add_argument("--initial", type=Path, required=True)
p.add_argument(
    "--methods",
    nargs="+",
    default=["poisson", "coverage", "random", "image_adols", "joint_adols"],
)
p.add_argument("--budgets", type=int, nargs="+", default=[32, 64, 128])
p.add_argument("--seeds", type=int, nargs="+", default=[42])
p.add_argument("--sensor-weight", type=float, default=1)
p.add_argument("--epochs", type=int, default=200)
a = p.parse_args()
a.upstream = a.upstream.resolve()
a.initial = a.initial.resolve()
env = os.environ.copy()
if Path("/usr/lib/wsl/lib").is_dir():
    env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib" + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
sensors = (
    ROOT.parent / "gaussian_observability/results/image_sensors_visual16.json"
).resolve()
env.update(
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
    PYTHONUNBUFFERED="1",
    GC_GAUSSIAN_CKPT=str(a.initial),
    GC_FRAMES="4",
    GC_SENSOR_FILE=str(sensors),
    GC_SENSOR_WEIGHT=str(a.sensor_weight),
    GC_EPOCHS=str(a.epochs),
)
logdir = ROOT / "logs"
logdir.mkdir(exist_ok=True)
for seed in a.seeds:
    for budget in a.budgets:
        for method in a.methods:
            tag = f"{method}_n{budget}_s{seed}_w{a.sensor_weight:g}_e{a.epochs}"
            output = ROOT / "results" / f"run_{tag}.json"
            if output.exists():
                print("Reuse completed", tag, flush=True)
                continue
            basis = ROOT / "results" / f"basis_{method}_{budget}.json"
            assert basis.exists()
            env.update(GC_RUN=tag, GC_SEED=str(seed), GC_BASIS_FILE=str(basis))
            before = set((a.upstream / "log/greencube").glob(f"basis_{tag}_*"))
            start = time.perf_counter()
            with (logdir / f"{tag}_train.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "main.py",
                        "--configs",
                        str(ROOT / "configs/selection.py"),
                        "--skip_visualize",
                        "--skip_render",
                    ],
                    cwd=a.upstream,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            train_seconds = time.perf_counter() - start
            runs = set((a.upstream / "log/greencube").glob(f"basis_{tag}_*")) - before
            assert len(runs) == 1
            run = runs.pop()
            evaluation = ROOT / "results" / f"eval_{tag}.json"
            with (logdir / f"{tag}_eval.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(
                            ROOT.parent
                            / "gaussian_observability/scripts/evaluate_images.py"
                        ),
                        "--upstream",
                        str(a.upstream),
                        "--run",
                        str(run),
                        "--config",
                        str(ROOT / "configs/selection.py"),
                        "--sensors",
                        str(sensors),
                        "--frames",
                        "4",
                        "--out",
                        str(evaluation),
                    ],
                    cwd=a.upstream,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            output.write_text(
                json.dumps(
                    {
                        "tag": tag,
                        "method": method,
                        "budget": budget,
                        "seed": seed,
                        "sensor_weight": a.sensor_weight,
                        "epochs": a.epochs,
                        "run_name": run.name,
                        "train_wall_seconds": train_seconds,
                        "evaluation_seconds": time.perf_counter()
                        - start
                        - train_seconds,
                        "basis_file": basis.name,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(
                "Completed", tag, "train seconds", round(train_seconds, 2), flush=True
            )
