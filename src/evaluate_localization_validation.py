"""Score UJI localization on its untouched publisher-provided validation file."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]


def main():
    model_path = ROOT / "models" / "trained" / "localization_uji.pkl"
    bundle = joblib.load(model_path)
    path = ROOT / "datasets" / "extracted" / "D1" / "UJIndoorLoc" / "validationData.csv"
    d = pd.read_csv(path)
    wap = bundle["wap_columns"]
    raw = d[wap].to_numpy(dtype=float)
    absent = raw == bundle["missing_sentinel"]
    signal = raw.copy()
    signal[absent] = bundle["missing_signal_fill"]
    X = np.hstack([signal, (~absent).astype(np.uint8)])
    actual = d[["LONGITUDE", "LATITUDE"]].to_numpy(dtype=float)
    pred = bundle["model"].predict(X)
    err = pred - actual
    distance = np.sqrt(np.sum(err ** 2, axis=1))
    metrics = {"n_validation": len(d),
               "mae_x": float(mean_absolute_error(actual[:, 0], pred[:, 0])),
               "mae_y": float(mean_absolute_error(actual[:, 1], pred[:, 1])),
               "rmse_x": float(np.sqrt(mean_squared_error(actual[:, 0], pred[:, 0]))),
               "rmse_y": float(np.sqrt(mean_squared_error(actual[:, 1], pred[:, 1]))),
               "median_euclidean_error_coordinate_units": float(np.median(distance)),
               "mean_euclidean_error_coordinate_units": float(np.mean(distance))}
    pd.DataFrame({"actual_x": actual[:, 0], "actual_y": actual[:, 1],
                  "predicted_x": pred[:, 0], "predicted_y": pred[:, 1],
                  "euclidean_error_coordinate_units": distance}).to_csv(
        ROOT / "models" / "comparison" / "localization_official_validation.csv", index=False)
    lines = ["# UJI Official Localization Validation\n\n",
             "The publisher-provided validation file was not used for fitting or model selection. The model uses the same 520 WAP channels and separate detection masks as in training.\n\n"]
    lines += [f"- {k}: {v}\n" for k, v in metrics.items()]
    lines.append("\nCoordinates are projected planar values with unconfirmed source CRS/origin; error is shown in coordinate units and cannot be read as metres until verified.\n")
    (ROOT / "reports" / "localization_official_validation.md").write_text("".join(lines), encoding="utf-8")
    print(metrics)


if __name__ == "__main__":
    main()
