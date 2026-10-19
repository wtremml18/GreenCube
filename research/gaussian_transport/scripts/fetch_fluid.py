"""Download checksum-pinned upstream archive and extract only selected volumes."""

from pathlib import Path
import argparse, hashlib, tarfile, urllib.request, json

ROOT = Path(__file__).resolve().parents[1]
URL = "https://huggingface.co/datasets/tnx123/LagrangianSplats-Dataset/resolve/main/gt/scalarsyn.tar.gz"
SHA = "48e1aece0b78e3cb65c9cf26c206971c39564e619d0a142abd98f45986ebf22d"
p = argparse.ArgumentParser()
p.add_argument("--archive", type=Path, default=ROOT / "data/scalarsyn.tar.gz")
args = p.parse_args()
args.archive.parent.mkdir(parents=True, exist_ok=True)
(ROOT / "data").mkdir(exist_ok=True)
if not args.archive.exists():
    urllib.request.urlretrieve(URL, args.archive)
with args.archive.open("rb") as f:
    actual = hashlib.file_digest(f, "sha256").hexdigest()
if actual != SHA:
    raise ValueError("Archive checksum mismatch")
wanted = {
    f"scalarsyn/{kind}_{frame:04d}.npz"
    for kind in ["density", "velocity"]
    for frame in [0, 10, 20, 29]
}
manifest = {"url": URL, "sha256": SHA, "frames": [0, 10, 20, 29], "files": []}
with tarfile.open(args.archive) as t:
    for member in t:
        if member.name in wanted:
            data = t.extractfile(member).read()
            out = ROOT / "data" / Path(member.name).name
            out.write_bytes(data)
            manifest["files"].append(
                {"name": out.name, "sha256": hashlib.sha256(data).hexdigest()}
            )
assert len(manifest["files"]) == 8
(ROOT / "data/fluid_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Verified and extracted 8 volumes")
