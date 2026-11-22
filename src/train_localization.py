"""Train a separate Wi-Fi fingerprint -> (x,y) indoor localization model."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "datasets" / "extracted" / "D1" / "UJIndoorLoc" / "trainingData.csv"
    d = pd.read_csv(source)
    wap = [c for c in d.columns if str(c).upper().startswith("WAP")]
    raw = d[wap].to_numpy(dtype=float)
    absent = raw == 100
    # -110 is only the numeric fill value; a parallel absent/present mask keeps
    # non-detection distinct from a measured signal of any dBm value.
    signal = raw.copy()
    signal[absent] = -110.0
    X = np.hstack([signal, (~absent).astype(np.uint8)])
    target = d[["LONGITUDE", "LATITUDE"]].to_numpy(dtype=float)
    groups = np.empty(len(d), dtype=object)
    for _, g in d.groupby("BUILDINGID"):
        bx = pd.cut(g["LONGITUDE"], bins=5, labels=False, duplicates="drop").fillna(0).astype(int)
        by = pd.cut(g["LATITUDE"], bins=5, labels=False, duplicates="drop").fillna(0).astype(int)
        groups[g.index.to_numpy()] = [f"{int(b)}_{int(x)}_{int(y)}"
                                      for b, x, y in zip(g.BUILDINGID, bx, by)]
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, groups=groups))
    model = RandomForestRegressor(n_estimators=100, max_depth=22, min_samples_leaf=2,
                                  max_features=.35, n_jobs=-1, random_state=42)
    model.fit(X[tr], target[tr])
    pred = model.predict(X[te])
    delta = pred - target[te]
    euclidean = np.sqrt(np.sum(delta ** 2, axis=1))
    metrics = {"n_train": int(len(tr)), "n_test": int(len(te)),
               "mae_x": float(mean_absolute_error(target[te, 0], pred[:, 0])),
               "mae_y": float(mean_absolute_error(target[te, 1], pred[:, 1])),
               "rmse_x": float(np.sqrt(mean_squared_error(target[te, 0], pred[:, 0]))),
               "rmse_y": float(np.sqrt(mean_squared_error(target[te, 1], pred[:, 1]))),
               "median_euclidean_error_coordinate_units": float(np.median(euclidean)),
               "mean_euclidean_error_coordinate_units": float(np.mean(euclidean))}
    out = ROOT / "models" / "trained" / "localization_uji.pkl"
    joblib.dump({"model": model, "wap_columns": wap, "missing_sentinel": 100,
                 "missing_signal_fill": -110, "features": "signal values + detection mask"}, out)
    pd.DataFrame({"actual_x": target[te, 0], "actual_y": target[te, 1],
                  "predicted_x": pred[:, 0], "predicted_y": pred[:, 1],
                  "euclidean_error_coordinate_units": euclidean}).to_csv(
        ROOT / "models" / "comparison" / "localization_test_predictions.csv", index=False)
    lines = ["# Indoor Localization Evaluation (separate objective)\n\n",
             "Model: Random Forest regressor maps the 520 UJI WAP readings to X/Y. ",
             "Missing APs are represented by a separate detection mask; +100 is not used as RSSI. ",
             "Whole 5x5 spatial cells per building were held out.\n\n",
             "## Spatial holdout metrics\n\n"]
    lines += [f"- {k}: {v}\n" for k, v in metrics.items()]
    lines += ["\nUJI coordinate columns are projected planar values, not GPS degrees. ",
              "The exact source CRS/origin is not established, so errors are reported in coordinate units; ",
              "these results do not transfer directly to the ESP32 local frame.\n"]
    (ROOT / "reports" / "localization.md").write_text("".join(lines), encoding="utf-8")
    print(metrics)


if __name__ == "__main__":
    main()
