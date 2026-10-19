"""Download only four public CSVs at a pinned upstream commit, verify hashes."""

from pathlib import Path
import hashlib, json, urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "826cd40a2f0827857012a06e08390190b58bfe54"


def main():
    records = []
    for dataset in ["gladfull", "laserfull"]:
        for prefix in ["XY", "UV"]:
            name = f"{prefix}_train_{dataset}.csv"
            url = f"https://raw.githubusercontent.com/renatoberlinghieri/Helmholtz-GP/{COMMIT}/data/{name}"
            p = ROOT / "data" / name
            if not p.exists():
                with urllib.request.urlopen(url, timeout=60) as r:
                    p.write_bytes(r.read())
            records.append(
                dict(
                    file=name,
                    url=url,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    bytes=p.stat().st_size,
                )
            )
    manifest = ROOT / "data" / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != records:
            raise RuntimeError("Data checksum mismatch")
    else:
        manifest.write_text(json.dumps(records, indent=2) + "\n")
    print("Verified four pinned public data files")


if __name__ == "__main__":
    main()
