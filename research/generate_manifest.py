"""Hash the non-ignored research artifacts without copying raw/private data."""

from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess

ROOT = Path(__file__).resolve().parent
repo = ROOT.parent
paths = subprocess.check_output(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "research"],
    cwd=repo,
    text=True,
).splitlines()
files = []
for name in sorted(set(paths)):
    p = repo / name
    if p.is_file() and p.name != "evidence_manifest.json":
        files.append(
            dict(
                path=name,
                bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            )
        )
manifest = dict(
    python=platform.python_version(),
    packages={
        name: importlib.metadata.version(name)
        for name in ["numpy", "scipy", "pandas", "matplotlib", "pytest"]
    },
    files=files,
)
(ROOT / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Hashed {len(files)} tracked-or-trackable research artifacts.")
