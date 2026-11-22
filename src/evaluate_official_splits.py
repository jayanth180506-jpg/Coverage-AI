"""Score D5 models on the untouched publisher-provided test split."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]


def metrics(actual, pred):
    return {"mae_dbm": float(mean_absolute_error(actual, pred)),
            "rmse_dbm": float(np.sqrt(mean_squared_error(actual, pred))),
            "r2": float(r2_score(actual, pred))}


def main():
    d = pd.read_csv(ROOT / "datasets" / "processed" / "D5_test_long.csv")
    bundle = joblib.load(ROOT / "models" / "trained" / "benchmark_model_bundle.pkl")
    ridge = joblib.load(ROOT / "models" / "trained" / "ridge_model.pkl")
    d["floor_id"] = pd.to_numeric(d.floor_id, errors="coerce").fillna(0).astype(int)
    d["building_num"] = d.building_id.map(bundle["building_map"])
    if d.building_num.isna().any():
        raise ValueError("Official D5 test contains an unseen building; define an explicit mapping before scoring")
    d["ap_mean_rssi"] = d.bssid.map(bundle["ap_mean_map"]).fillna(bundle["global_mean_rssi"])
    feat = bundle["feature_names"]
    results = []
    for name, model in [("ridge", ridge), ("decision_tree", bundle["tree"])]:
        pred = model.predict(d[feat])
        row = {"model": name, "n": len(d), **metrics(d.rssi, pred),
               "unknown_ap_fraction": float((~d.bssid.isin(bundle["ap_mean_map"])).mean())}
        results.append(row)
        d[f"{name}_predicted_rssi"] = pred
    pd.DataFrame(results).to_csv(ROOT / "models" / "comparison" / "D5_official_test_metrics.csv", index=False)
    d.to_csv(ROOT / "models" / "comparison" / "D5_official_test_predictions.csv", index=False)
    lines = ["# D5 Official Test Evaluation\n\n",
             "This publisher-provided split was not used for feature selection or model tuning. Models were fitted using the D5 training file, with a spatially held-out validation region used during training. AP target means use training rows only.\n\n",
             pd.DataFrame(results).to_markdown(index=False),
             "\n\nD5 official test rows are relatively few; interpret these metrics alongside the larger spatial holdout results.\n"]
    (ROOT / "reports" / "D5_official_test.md").write_text("".join(lines), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
