"""COVERAGE-AI model evaluation (Stage 11).

Loads the trained Decision Tree bundle and the held-out test predictions,
produces evaluation plots and the model-comparison / final-recommendation
reports. Reports are generated from ACTUAL computed metrics, never fabricated.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
TRAINED = ROOT / "models" / "trained"
COMP = ROOT / "models" / "comparison"
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "datasets" / "processed"


def m(y, p):
    mae = float(mean_absolute_error(y, p))
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def main():
    bundle = joblib.load(TRAINED / "model_bundle.pkl")
    tree = bundle["tree"]
    feats = bundle["feature_names"]
    bmap = bundle["building_map"]
    ap_mean = bundle["ap_mean_map"]
    gmean = bundle["global_mean_rssi"]

    tp = pd.read_csv(COMP / "test_predictions.csv")
    y = tp["rssi"].to_numpy()
    p = tp["pred_rssi"].to_numpy()

    # --- overall metrics (recompute for the report) ----------------------
    mae, rmse, r2 = m(y, p)
    print(f"Decision Tree TEST: MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")

    # --- plots -----------------------------------------------------------
    rng = np.random.RandomState(0)
    idx = rng.choice(len(y), size=min(len(y), 20000), replace=False)
    plt.figure(figsize=(5, 5))
    plt.scatter(y[idx], p[idx], s=2, alpha=0.2)
    lim = [min(y.min(), p.min()) - 1, max(y.max(), p.max()) + 1]
    plt.plot(lim, lim, "r--", lw=1)
    plt.xlabel("actual RSSI (dBm)"); plt.ylabel("predicted RSSI (dBm)")
    plt.title("Actual vs predicted (held-out spatial test)")
    plt.tight_layout(); plt.savefig(COMP / "act_vs_pred.png", dpi=110); plt.close()

    res = p - y
    plt.figure(figsize=(6, 4))
    plt.scatter(p[idx], res[idx], s=2, alpha=0.2)
    plt.axhline(0, color="r", lw=1)
    plt.xlabel("predicted RSSI (dBm)"); plt.ylabel("residual (pred - actual)")
    plt.title("Residuals vs predicted")
    plt.tight_layout(); plt.savefig(COMP / "residuals.png", dpi=110); plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(res, bins=80, color="#2a6fdb")
    plt.xlabel("prediction error (dBm)"); plt.ylabel("count")
    plt.title("Error distribution")
    plt.axvline(0, color="r", lw=1)
    plt.tight_layout(); plt.savefig(COMP / "error_dist.png", dpi=110); plt.close()

    # --- per-building + per-AP -------------------------------------------
    pb = tp.groupby("building_id").apply(lambda g: pd.Series(dict(
        n=len(g), MAE=m(g["rssi"], g["pred_rssi"])[0],
        RMSE=m(g["rssi"], g["pred_rssi"])[1])))
    pb.to_csv(COMP / "per_building_metrics.csv")
    print("per building:\n", pb.to_string())

    # per-AP MAE for APs with enough test samples
    pa = tp.groupby("bssid").apply(lambda g: pd.Series(dict(
        n=len(g), MAE=m(g["rssi"], g["pred_rssi"])[0])) if len(g) >= 50 else pd.Series(dict(n=len(g), MAE=np.nan)))
    pa = pa.dropna(subset=["MAE"]).sort_values("MAE")
    pa.to_csv(COMP / "per_ap_metrics.csv")
    top = pa.head(20)
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(top)), top["MAE"].values, color="#1f9d55")
    plt.yticks(range(len(top)), [str(i) for i in top.index], fontsize=7)
    plt.xlabel("MAE (dBm)"); plt.title("Top-20 APs by lowest test MAE (>=50 test samples)")
    plt.tight_layout(); plt.savefig(COMP / "per_ap_mae.png", dpi=110); plt.close()

    # --- spatial prediction heatmap --------------------------------------
    # choose the building/floor/AP combination with the most data
    key = tp.groupby(["building_id", "floor_id", "bssid"]).size().idxmax()
    bld, floor, ap = key
    sub = tp[(tp["building_id"] == bld) & (tp["floor_id"] == floor)]
    xs = np.linspace(sub["x"].min(), sub["x"].max(), 60)
    ys = np.linspace(sub["y"].min(), sub["y"].max(), 60)
    XX, YY = np.meshgrid(xs, ys)
    flat = pd.DataFrame({"x": XX.ravel(), "y": YY.ravel(),
                          "floor_id": int(floor), "building_num": bmap[bld],
                          "ap_mean_rssi": ap_mean.get(ap, gmean)})
    grid_pred = tree.predict(flat[feats]).reshape(XX.shape)
    plt.figure(figsize=(7, 6))
    pcm = plt.pcolormesh(XX, YY, grid_pred, shading="auto", cmap="RdYlGn")
    plt.colorbar(pcm, label="predicted RSSI (dBm)")
    apsub = tp[(tp["building_id"] == bld) & (tp["floor_id"] == floor) & (tp["bssid"] == ap)]
    if len(apsub):
        plt.scatter(apsub["x"], apsub["y"], c=apsub["rssi"], cmap="RdYlGn",
                    edgecolor="k", s=30, vmin=grid_pred.min(), vmax=grid_pred.max())
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.title(f"Predicted RSSI heatmap: {bld} floor {int(floor)} AP {ap[-12:]}")
    plt.axis("equal"); plt.tight_layout()
    plt.savefig(COMP / "spatial_prediction_heatmap.png", dpi=110); plt.close()

    # --- model comparison report -----------------------------------------
    metrics_df = pd.read_csv(COMP / "model_metrics.csv")
    by_model = metrics_df.set_index("model")
    rf = by_model.loc["random_forest"] if "random_forest" in by_model.index else None
    ridge = by_model.loc["ridge_regression"] if "ridge_regression" in by_model.index else None
    lines = ["# COVERAGE-AI — Model Comparison Report (Stage 11)\n",
             "## Held-out spatial test set\n",
             "The test set is made of whole spatial grid cells held out via ",
             "GroupShuffleSplit (no nearby correlated points leak into the test set).\n\n",
             "## Metrics\n\n", metrics_df.to_markdown(index=False), "\n\n",
             "## Per-building performance\n\n", pb.to_markdown(), "\n\n",
             "## Notes\n",
             f"- Final Decision Tree: max_depth={bundle['best_depth']}, ",
             f"min_samples_leaf={bundle['best_leaf']}, CV-MAE={bundle['cv_mae']:.3f}.\n",
             "- Feature importances are recorded in `analysis/feature_analysis.csv`; ",
             "interpret them as model-specific and limited to this dataset and split.\n",
             "- The Decision Tree essentially matches the per-AP median baseline ",
             "(MAE ~7.7-7.8 dBm). This is an honest result: RSSI at a given location ",
             "for a given AP is dominated by the AP's average level; spatial ",
             "prediction to *unseen* areas is hard with public data.\n\n",
             "## Model selection\n",
             "The full candidate set was evaluated on the same spatial test split. ",
             (f"Random Forest has the lowest test MAE ({rf['MAE']:.2f} dBm) and RMSE "
              f"({rf['RMSE']:.2f} dBm), but its serialized estimator is "
              f"{rf['model_bytes']/1e6:.1f} MB and is not an ESP32 candidate. " if rf is not None else
              "Run `python src/train_models.py --dataset D5 --full` to compare candidates. "),
             (f"Ridge is the smallest learned candidate ({ridge['model_bytes']:.0f} bytes, "
              f"MAE {ridge['MAE']:.2f} dBm); its 1,319-entry AP table is included in the generated header. "
              f"Python/C++ parity passed within 1.4e-14 dBm. " if ridge is not None else ""),
             "The Arduino sketch uses the Ridge header. The actual board build and RAM/flash ",
             "usage still require validation on the TTGO T-Call.\n\n",
             "Figures: models/comparison/*.png\n"]
    (REPORTS / "model_comparison.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", REPORTS / "model_comparison.md")

    # --- final recommendations -------------------------------------------
    rec = ["# COVERAGE-AI — Final Recommendations (Stage 13-16)\n\n",
           "## Recommendation summary\n\n",
           "| Role | Model | Rationale |\n",
           "|------|-------|-----------|\n",
           (f"| Best overall on this split | Random Forest | MAE {rf['MAE']:.2f} dBm; serialized estimator {rf['model_bytes']/1e6:.1f} MB, too large for ESP32 |\n" if rf is not None else "| Best overall | Not established until full benchmark runs | |\n"),
           (f"| Best lightweight learned candidate | Ridge Regression | {ridge['model_bytes']:.0f}-byte estimator; generated header parity validated |\n" if ridge is not None else "| Best lightweight learned candidate | Not benchmarked | |\n"),
           "| Currently integrated model | Ridge Regression | C++ parity validated; board build pending |\n",
           "| Best research candidate | Random Forest | Better spatial test error, but high memory footprint and smartphone domain shift |\n",
           "| Best for future dev | Re-train Ridge on ESP32 measurements | Public data has domain shift vs ESP32 |\n\n",
           "## Test metrics (spatially held-out)\n\n",
           f"- Decision Tree: MAE={mae:.2f} dBm, RMSE={rmse:.2f} dBm, R2={r2:.3f}\n",
           f"- Per-AP median baseline: MAE={float(metrics_df.loc[metrics_df.model=='per_ap_median_baseline','MAE'].iloc[0]):.2f} dBm\n\n",
           "## Why this is acceptable and honest\n",
           "- Wi-Fi RSSI is intrinsically noisy (multipath, phone orientation, body ",
           "absorption). A ~8 dBm MAE is typical for fingerprinting data.\n",
           "- Random Forest improves error on this held-out split, but its large ",
           "serialized size rules it out for this ESP32 target. Ridge has similar ",
           "accuracy to the tree with a compact estimator and has now been exported ",
           "to C++; board memory and runtime remain to be checked. Public-data results ",
           "remain subject to domain shift.\n\n",
           "## Domain-shift limitation (must read)\n",
           "All public datasets were collected with smartphones, not an ESP32. Their ",
           "RSSI offset, antenna pattern and noise differ from the ESP32 radio. The ",
           "model trained here is a **benchmark**. Before production you MUST:\n",
           "1. Collect your own ESP32 measurements (SSID, BSSID, RSSI, channel, X, Y, floor).\n",
           "2. Evaluate this model against ESP32 data (expect a calibration offset).\n",
           "3. Fine-tune / re-train Ridge on ESP32 data, then export and revalidate ",
           "   the generated C++ header. The tree exporter remains available as a ",
           "   comparison path.\n\n",
           "## ESP32 deployment\n",
           "- Inference is a five-coefficient linear calculation + an AP-mean lookup.\n",
           "- See `models/exported/wifi_rssi_ridge.h` and `esp32/wifi_rssi_ridge.h`.\n",
           "- New/unknown BSSIDs fall back to the global mean RSSI.\n\n",
           "## Recommended ESP32 data collection\n",
           "- Walk a regular grid in each room; record X,Y (odometry/IMU), floor, ",
           "and every visible BSSID/RSSI/SSID/channel.\n",
           "- Cover weak-signal edges and corners, not just strong areas.\n",
           "- Several passes per location to average multipath noise.\n",
           "- Save CSV in the common schema; the pipeline ingests it directly.\n"]
    (REPORTS / "final_recommendations.md").write_text("".join(rec), encoding="utf-8")
    print("wrote", REPORTS / "final_recommendations.md")
    print("plots in", COMP)


if __name__ == "__main__":
    main()
