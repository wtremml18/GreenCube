"""Fetch and inspect public Lynch-group inputs; does not fit a reconstruction."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np
from scipy.io import loadmat
from cdflib import CDF, cdfepoch
import h5py

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "sources.json").read_text())
    (ROOT / "data").mkdir(exist_ok=True)
    for item in manifest:
        path = ROOT / "data" / item["file"]
        if not path.exists():
            if not args.fetch:
                raise FileNotFoundError(f"{path.name}: rerun with --fetch")
            urllib.request.urlretrieve(item["url"], path)
        with path.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"Changed source bytes: {item['file']}")
    flow = loadmat(ROOT / "data/flow_data_ig.mat", simplify_cells=True)
    numeric = [
        "time",
        "lat_100km",
        "lon_100km",
        "mlat",
        "mlon",
        "v_geog_east",
        "v_geog_north",
        "v_geom_east",
        "v_geom_north",
    ]
    assert all(flow[k].shape == (195,) for k in numeric)
    audit = {
        "scope": "Data availability, schema and preliminary quality inspection only; no auroral reconstruction validated.",
        "isinglass": {
            "samples": 195,
            "strictly_increasing_time": bool(np.all(np.diff(flow["time"]) > 0)),
            "numeric_columns": {
                k: {
                    "finite": int(np.isfinite(flow[k]).sum()),
                    "min": float(np.nanmin(flow[k])),
                    "max": float(np.nanmax(flow[k])),
                }
                for k in numeric
            },
            "uncertainty": "No numerical measurement-error array among inspected columns. MATLAB datetime/comment objects are not decoded by this loader.",
        },
        "swarm": {},
    }
    for sat in ["a", "c"]:
        c = CDF(ROOT / "data" / f"swarm_{sat}_20230210.cdf")
        times = cdfepoch.to_datetime(c.varget("Timestamp"))
        keep = np.abs(times - np.datetime64("2023-02-10T09:51:27")) <= np.timedelta64(
            120, "s"
        )
        flags = c.varget("Quality_flags")[keep]
        n = int(keep.sum())
        audit["swarm"][sat.upper()] = {
            "records": len(times),
            "first_utc": str(times[0]),
            "last_utc": str(times[-1]),
            "window": "2023-02-10T09:51:27Z +/-120 s; time selection only, not a spatial conjunction mask",
            "window_records": n,
            "quality_bit_pass_counts": {
                v: int(np.sum((flags & (1 << j)) != 0))
                for j, v in enumerate(["Vixh", "Vixv", "Viy", "Viz"])
            },
            "variables": c.cdf_info().zVariables,
            "velocity_frame": c.varattsget("Viy")["CATDESC"],
            "velocity_units": c.varattsget("Viy")["UNITS"],
            "interpretation": "Bit pass is one documented quality criterion, not a complete validity mask; release notes/calibration and error estimates still require review.",
        }
        assert n == 480
    with h5py.File(ROOT / "data/pfisr_20230210.h5") as f:
        wanted = [
            "Time/UnixTime",
            "VvelsMagCoords/Velocity",
            "VvelsMagCoords/CovarianceV",
            "VvelsGeoCoords/Velocity",
            "VvelsGeoCoords/CovarianceV",
            "VvelsGeoCoords/ElectricField",
            "VvelsGeoCoords/CovarianceE",
        ]
        audit["pfisr"] = {
            k: {
                "shape": list(f[k].shape),
                "units": f[k].attrs.get("Units", b"").decode(),
                "title": f[k].attrs.get("TITLE", b"").decode(),
            }
            for k in wanted
        }
        assert f["VvelsMagCoords/Velocity"].shape == (141, 20, 3)
        assert f["VvelsMagCoords/CovarianceV"].shape == (141, 20, 3, 3)
    lines = (ROOT / "data/events.dat").read_text().splitlines()
    events = []
    for line in lines:
        cols = line.split()
        if len(cols) > 3 and cols[0].isdigit() and "T" in cols[1]:
            events.append({"id": cols[0], "utc": cols[1], "spacecraft": cols[2]})
    audit["events"] = {
        "spacecraft_event_rows": len(events),
        "distinct_conjunction_times": len({r["utc"] for r in events}),
        "rows": events,
    }
    assert len(events) == 8 and audit["events"]["distinct_conjunction_times"] == 6
    with h5py.File(ROOT / "data/boundaries.mat") as f:
        audit["boundaries"] = {
            "top_level_keys": list(f.keys()),
            "format": "MATLAB v7.3/HDF5 with object references; a schema listing is not a decoded physical boundary.",
        }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/schema_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        "Verified six source hashes; inspected 195 rocket samples, two Swarm files, PFISR covariance schema, and six conjunction times."
    )


if __name__ == "__main__":
    main()
