"""Fit a small Ridge regressor on D4's official training file and score its test file."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    train = pd.read_csv(ROOT / "datasets" / "processed" / "D4_training_long.csv")
    test = pd.read_csv(ROOT / "datasets" / "processed" / "D4_test_long.csv")
    apmeans = train.groupby("bssid").rssi.mean()
    global_mean = float(train.rssi.mean())
    for d in [train, test]:
        d["ap_mean_rssi"] = d.bssid.map(apmeans).fillna(global_mean)
    features = ["x", "y", "ap_mean_rssi"]
    model = Ridge(alpha=1.0).fit(train[features], train.rssi)
    pred = model.predict(test[features])
    metrics = {"n_train": len(train), "n_official_test": len(test),
               "mae_dbm": float(mean_absolute_error(test.rssi, pred)),
               "rmse_dbm": float(np.sqrt(mean_squared_error(test.rssi, pred))),
               "r2": float(r2_score(test.rssi, pred)),
               "unknown_ap_fraction": float((~test.bssid.isin(apmeans.index)).mean())}
    pd.DataFrame([metrics]).to_csv(ROOT / "models" / "comparison" / "D4_official_test_metrics.csv", index=False)
    pd.DataFrame({"actual_rssi": test.rssi, "predicted_rssi": pred,
                  "x": test.x, "y": test.y, "bssid": test.bssid}).to_csv(
        ROOT / "models" / "comparison" / "D4_official_test_predictions.csv", index=False)
    (ROOT / "reports" / "D4_official_test.md").write_text(
        "# D4 Official Test Evaluation\n\n"
        "A separate Ridge regressor was fit only on D4's official training data. Coordinates and AP means use D4's own source frame. No D5 model parameters or D4 test rows were used in fitting.\n\n" +
        "\n".join(f"- {k}: {v}" for k, v in metrics.items()) +
        "\n\nThis dataset-specific benchmark is not an ESP32 validation.\n", encoding="utf-8")
    print(metrics)


if __name__ == "__main__":
    main()
