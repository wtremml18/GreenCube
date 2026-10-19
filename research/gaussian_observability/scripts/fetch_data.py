"""Acquire checksum-pinned ScalarSyn cameras and density/velocity volumes."""

import argparse, hashlib, tarfile, urllib.request, json
from pathlib import Path

SOURCES = [
    (
        "images",
        "https://huggingface.co/datasets/tnx123/LagrangianSplats-Dataset/resolve/main/data/scalarsyn.tar.gz",
        "26460ff1e8bb780b7ece48de05880c06d5e0332621c7a966bdad2320f14739bd",
        "data/smoke",
    ),
    (
        "ground_truth",
        "https://huggingface.co/datasets/tnx123/LagrangianSplats-Dataset/resolve/main/gt/scalarsyn.tar.gz",
        "48e1aece0b78e3cb65c9cf26c206971c39564e619d0a142abd98f45986ebf22d",
        "gt",
    ),
]
p = argparse.ArgumentParser()
p.add_argument("--upstream", type=Path, required=True)
p.add_argument("--cache", type=Path, required=True)
a = p.parse_args()
a.cache.mkdir(exist_ok=True, parents=True)
manifest = []
for name, url, sha, destination in SOURCES:
    archive = a.cache / f"scalarsyn_{name}.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(url, archive)
    digest = hashlib.sha256()
    with archive.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != sha:
        raise ValueError(f"Checksum mismatch: {archive}")
    out = a.upstream / destination
    out.mkdir(exist_ok=True, parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(out, filter="data")
    manifest.append(dict(name=name, url=url, sha256=sha, bytes=archive.stat().st_size))
    print("Verified and extracted", name, flush=True)
(Path(__file__).resolve().parents[1] / "results/sources.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
