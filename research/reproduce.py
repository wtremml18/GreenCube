"""Run the complete CPU research pipeline from any working directory."""

from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
stages = [
    ("track_observability", "fetch_data.py", []),
    ("track_observability", "run_synthetic.py", []),
    ("track_observability", "run_noise_audit.py", []),
    ("track_observability", "run_design.py", []),
    ("track_observability", "run_public.py", []),
    ("track_observability", "run_baseline_audits.py", []),
    ("track_observability", "run_robustness.py", []),
    ("track_observability", "run_geometry.py", []),
    ("track_observability", "summarize.py", []),
    ("gaussian_transport", "fetch_fluid.py", []),
    ("gaussian_transport", "run_benchmark.py", []),
    ("gaussian_transport", "run_ambiguity.py", []),
    ("gaussian_transport", "run_fluid.py", []),
    ("gaussian_transport", "run_fluid.py", ["--support-matched"]),
    ("gaussian_transport", "summarize.py", []),
]
env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
for folder, script, args in stages:
    print(f"Running {folder}/{script} {' '.join(args)}", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / folder / "scripts" / script), *args],
        check=True,
        env=env,
    )
subprocess.run([sys.executable, str(ROOT / "validate_results.py")], check=True, env=env)
print("All research stages completed.")
