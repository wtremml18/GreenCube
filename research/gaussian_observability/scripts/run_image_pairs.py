"""Sequential paired camera/synthetic-sensor runs in a prepared upstream checkout."""

import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--upstream", type=Path, required=True)
p.add_argument(
    "--initial", type=Path, required=True, help="Image-only frame-zero model directory"
)
p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
p.add_argument("--weight", type=float, default=0.01)
p.add_argument("--sensors", type=Path, default=ROOT / "results/image_sensors16.json")
p.add_argument("--label", default="sensors16")
p.add_argument("--reuse-baselines", action="store_true")
p.add_argument("--frames", type=int, default=4)
a = p.parse_args()
a.upstream = a.upstream.resolve()
a.initial = a.initial.resolve()
a.sensors = a.sensors.resolve()
env = os.environ.copy()
if Path("/usr/lib/wsl/lib").is_dir():
    env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib" + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
env.update(
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
    PYTHONUNBUFFERED="1",
    GC_GAUSSIAN_CKPT=str(a.initial),
    GC_FRAMES=str(a.frames),
)
logdir = ROOT / "logs"
logdir.mkdir(exist_ok=True)
manifest = []
if a.reuse_baselines and a.frames != 4:
    raise ValueError("Saved reusable baselines contain exactly four frames")
base_label = "images" if a.label == "sensors16" else "images-" + a.label
for seed in a.seeds:
    for condition in [base_label, a.label]:
        tag = f"{condition}_s{seed}_w{a.weight:g}"
        env.update(
            GC_RUN=tag,
            GC_SEED=str(seed),
            GC_SENSOR_FILE=str(a.sensors),
            GC_SENSOR_WEIGHT=str(a.weight if condition == a.label else 0),
        )
        start = time.time()
        if condition == base_label and a.reuse_baselines:
            previous = json.loads(
                (ROOT / "results" / f"images_images_s{seed}_w0.01.json").read_text()
            )
            run = a.upstream / "log/greencube" / previous["run_name"]
        else:
            before = set((a.upstream / "log/greencube").glob(f"images_{tag}_*"))
            with (logdir / f"{tag}_train.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "main.py",
                        "--configs",
                        str(ROOT / "configs/scalarsyn_images.py"),
                        "--skip_visualize",
                        "--skip_render",
                    ],
                    cwd=a.upstream,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            new = set((a.upstream / "log/greencube").glob(f"images_{tag}_*")) - before
            assert len(new) == 1, new
            run = new.pop()
        with (logdir / f"{tag}_eval.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/evaluate_images.py"),
                    "--upstream",
                    str(a.upstream),
                    "--run",
                    str(run),
                    "--config",
                    str(ROOT / "configs/scalarsyn_images.py"),
                    "--sensors",
                    str(a.sensors),
                    "--frames",
                    str(a.frames),
                    "--out",
                    str(ROOT / "results" / f"images_{tag}.json"),
                ],
                cwd=a.upstream,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        manifest.append(
            dict(
                seed=seed,
                condition=condition,
                weight=a.weight,
                run_name=run.name,
                sensor_file=a.sensors.name,
                elapsed_seconds=time.time() - start,
            )
        )
        (ROOT / "results" / f"image_runs_{a.label}_w{a.weight:g}.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(tag, "complete", round(time.time() - start, 1), flush=True)
