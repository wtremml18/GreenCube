"""Essential comparator: conventional RBF regression using only instrument values.

Sensor positions may be camera-informed, but no temporal image loss or splat
motion is used. Hyperparameters use only CV on the 16 supplied sensor values.
"""

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "track_observability"))
from src.kernels import tune, folds

samples = np.load(ROOT / "results/common_samples.npz")
rows = []
predictions = {}
for path in sorted((ROOT / "results").glob("image_sensors*.json")):
    d = json.loads(path.read_text())
    for frame in sorted(int(k) for k in d if k.isdigit()):
        x = np.array(d[str(frame)]["xyz"]) / 100
        y = np.array(d[str(frame)]["velocity"])
        q = samples[f"indices_frame{frame}"][:, ::-1] / 100
        truth = samples[f"truth_frame{frame}"]
        for kind in ["df", "independent"]:
            model, chosen, _ = tune(
                x,
                y,
                folds(len(x), seed=0),
                kind,
                lengths=(0.08, 0.16, 0.32, 0.64),
                ridges=(0.001, 0.03, 0.3),
            )
            prediction = model.predict(q)
            rows.append(
                dict(
                    layout=path.stem,
                    frame=frame,
                    kind=kind,
                    vector_rmse=float(
                        np.sqrt(np.mean(np.sum((prediction - truth) ** 2, axis=1)))
                    ),
                    constant_rmse=float(
                        np.sqrt(np.mean(np.sum((y.mean(0) - truth) ** 2, axis=1)))
                    ),
                    **chosen,
                )
            )
            predictions[f"{path.stem}_{kind}_frame{frame}"] = prediction
pd.DataFrame(rows).to_csv(ROOT / "results/sensor_only.csv", index=False)
np.savez_compressed(ROOT / "results/sensor_only_predictions.npz", **predictions)
print(pd.DataFrame(rows).groupby(["layout", "kind"]).vector_rmse.mean().to_string())
