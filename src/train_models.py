"""COVERAGE-AI model training (Stages 7-11).

Primary objective: predict Wi-Fi RSSI (dBm) at a physical location for a given
access point, using only features available on the ESP32 at inference time:

    features = [x, y, floor_id, building_num, ap_mean_rssi]

Design decisions (see spec):
  * BSSID is NOT encoded as an ordinal integer. It is TARGET-ENCODED: each AP
    is represented by its mean RSSI computed on the training fold only (one
    compact continuous feature + a small lookup table for deployment). Unseen
    APs fall back to the global mean. This is embeddable on an ESP32.
  * Spatially separated split: the (x,y) plane of each building is binned into a
    grid; whole spatial cells are held out via GroupShuffleSplit so the test
    set is genuinely unseen *areas*, not nearby correlated points.
  * Per-AP Decision Trees are also trained for the top-N measured APs as an
    honest comparison (no AP encoding needed).
  * Decision Tree is the PRIMARY model. Median baseline is included for
    context. Other regressors are NOT trained unless --full is passed.

Outputs:
  models/trained/model_bundle.pkl   (final tree + AP-mean map + meta)
  models/comparison/model_metrics.csv
  analysis/feature_analysis.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import time
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "datasets" / "processed"
TRAINED = ROOT / "models" / "trained"
COMPARISON = ROOT / "models" / "comparison"
ANALYSIS = ROOT / "analysis"
TRAINED.mkdir(parents=True, exist_ok=True)
COMPARISON.mkdir(parents=True, exist_ok=True)
ANALYSIS.mkdir(parents=True, exist_ok=True)

FEATURES = ["x", "y", "floor_id", "building_num", "ap_mean_rssi"]
TARGET = "rssi"


def load_primary(dataset="D5"):
    p = PROCESSED / f"{dataset}_long.csv"
    df = pd.read_csv(p)
    # z is auxiliary; keep floor_id as int category
    if "z" in df.columns:
        df = df.drop(columns=["z"])
    df["floor_id"] = df["floor_id"].astype(int)
    return df


def add_spatial_cells(df, n_bins=5):
    """Bin (x,y) per building into a grid; each cell is a group for spatial
    holdout so test areas are contiguous and separated from train."""
    df = df.copy()
    df["cell_id"] = ""
    for bld, g in df.groupby("building_id"):
        x_edges = np.linspace(g["x"].min() - 1e-6, g["x"].max() + 1e-6, n_bins + 1)
        y_edges = np.linspace(g["y"].min() - 1e-6, g["y"].max() + 1e-6, n_bins + 1)
        df.loc[g.index, "cx"] = np.digitize(g["x"].to_numpy(), x_edges[1:-1])
        df.loc[g.index, "cy"] = np.digitize(g["y"].to_numpy(), y_edges[1:-1])
    df["cell_id"] = (df["building_id"].astype(str) + "_" +
                     df["cx"].astype(int).astype(str) + "_" +
                     df["cy"].astype(int).astype(str))
    return df


def encode_ap_mean(df_fit, df_apply):
    """Target-encode bssid using the mean RSSI of each AP computed on df_fit.
    Unseen APs fall back to the global mean of df_fit."""
    ap_mean = df_fit.groupby("bssid")[TARGET].mean()
    global_mean = df_fit[TARGET].mean()
    return df_apply["bssid"].map(ap_mean).fillna(global_mean).astype(float), ap_mean, global_mean


def building_map_of(df):
    return {b: i for i, b in enumerate(sorted(df["building_id"].unique()))}


def make_X(df, bmap):
    X = pd.DataFrame({
        "x": df["x"].astype(float),
        "y": df["y"].astype(float),
        "floor_id": df["floor_id"].astype(int),
        "building_num": df["building_id"].map(bmap).astype(int),
    })
    return X


def metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(MAE=mae, RMSE=rmse, R2=r2)


def cv_select_depth(df_train, bmap, depths, leaf):
    """GroupKFold (spatial cells) CV to select max_depth + min_samples_leaf.
    AP target encoding is recomputed inside each fold to avoid leakage."""
    cells = df_train["cell_id"].to_numpy()
    best = None
    for d in depths:
        for lf in leaf:
            maes = []
            gkf = GroupKFold(n_splits=3)
            for tr_idx, va_idx in gkf.split(df_train, groups=cells):
                ftr = df_train.iloc[tr_idx]
                fva = df_train.iloc[va_idx]
                am, _, _ = encode_ap_mean(ftr, fva)
                Xtr = make_X(ftr, bmap); Xtr["ap_mean_rssi"] = am
                Xva = make_X(fva, bmap); Xva["ap_mean_rssi"] = encode_ap_mean(ftr, fva)[0]
                m = DecisionTreeRegressor(max_depth=d, min_samples_leaf=lf, random_state=42)
                m.fit(Xtr[FEATURES], ftr[TARGET])
                maes.append(metrics(fva[TARGET].to_numpy(), m.predict(Xva[FEATURES]))["MAE"])
            mean_mae = float(np.mean(maes))
            if best is None or mean_mae < best["mae"]:
                best = dict(depth=d, leaf=lf, mae=mean_mae)
            print(f"  CV depth={d} leaf={lf}: MAE={mean_mae:.3f}")
    return best


def per_ap_trees(df_train, df_test, bmap, top_n=15):
    """Train one DecisionTree per top-N AP (features x,y,floor only). Report
    mean test MAE across APs that appear in BOTH train and test."""
    counts = df_train["bssid"].value_counts()
    top = counts.head(top_n).index.tolist()
    rows = []
    for ap in top:
        tr = df_train[df_train["bssid"] == ap]
        te = df_test[df_test["bssid"] == ap]
        if len(te) < 20:
            continue
        Xtr = tr[["x", "y", "floor_id"]]
        Xte = te[["x", "y", "floor_id"]]
        m = DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42)
        m.fit(Xtr, tr[TARGET])
        mm = metrics(te[TARGET].to_numpy(), m.predict(Xte))
        rows.append(dict(model="per_ap_tree", ap=ap, n_train=len(tr), n_test=len(te),
                         MAE=mm["MAE"], RMSE=mm["RMSE"], R2=mm["R2"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="D5")
    ap.add_argument("--full", action="store_true",
                    help="also train Linear/Ridge/RF/GBR for comparison")
    args = ap.parse_args()

    print(f"Loading {args.dataset} long format ...")
    df = load_primary(args.dataset)
    print(f"  rows={len(df)} APs={df['bssid'].nunique()} buildings={sorted(df['building_id'].unique().tolist())}")

    df = add_spatial_cells(df, n_bins=5)
    bmap = building_map_of(df)

    # --- spatial split: hold out whole spatial cells -----------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    trainval_idx, test_idx = next(gss.split(df, groups=df["cell_id"]))
    df_tv = df.iloc[trainval_idx].copy()
    df_test = df.iloc[test_idx].copy()
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, val_idx = next(gss2.split(df_tv, groups=df_tv["cell_id"]))
    df_train = df_tv.iloc[tr_idx].copy()
    df_val = df_tv.iloc[val_idx].copy()
    print(f"  spatial split -> train={len(df_train)} val={len(df_val)} test={len(df_test)}")
    print(f"  train cells={df_train['cell_id'].nunique()} test cells={df_test['cell_id'].nunique()}")

    # --- honest baselines: global mean/median and per-AP mean/median --------
    # Match the final tree/candidate fitting set (train + validation); the test
    # set remains untouched. Depth selection still uses train-only CV above.
    global_mean = float(df_tv[TARGET].mean())
    gmed = float(df_tv[TARGET].median())
    ap_mean = df_tv.groupby("bssid")[TARGET].mean()
    ap_med = df_tv.groupby("bssid")[TARGET].median()
    baseline_metrics = {}
    for name, values, fallback in [
        ("global_mean_baseline", None, global_mean),
        ("global_median_baseline", None, gmed),
        ("per_ap_mean_baseline", ap_mean, global_mean),
        ("per_ap_median_baseline", ap_med, gmed),
    ]:
        pred0 = (np.full(len(df_test), fallback) if values is None
                 else df_test["bssid"].map(values).fillna(fallback).to_numpy())
        baseline_metrics[name] = metrics(df_test[TARGET].to_numpy(), pred0)
        print(f"  {name}: {baseline_metrics[name]}")

    # --- Decision Tree: CV-select depth -----------------------------------
    print("Selecting Decision Tree depth via spatial GroupKFold CV ...")
    best = cv_select_depth(df_train, bmap,
                           depths=[4, 6, 8, 10, 12],
                           leaf=[20, 50])
    print(f"  best: depth={best['depth']} leaf={best['leaf']} CV_MAE={best['mae']:.3f}")

    # --- final model: train on train+val (trainval) at best depth ----------
    am_tv, ap_mean_final, gmean_final = encode_ap_mean(df_tv, df_tv)
    df_tv["ap_mean_rssi"] = am_tv
    X_tv = make_X(df_tv, bmap); X_tv["ap_mean_rssi"] = df_tv["ap_mean_rssi"]
    tree = DecisionTreeRegressor(max_depth=best["depth"], min_samples_leaf=best["leaf"],
                                 random_state=42)
    tree.fit(X_tv[FEATURES], df_tv[TARGET])
    # encode test with trainval AP stats (test unseen)
    df_test["ap_mean_rssi"] = encode_ap_mean(df_tv, df_test)[0]
    X_test = make_X(df_test, bmap); X_test["ap_mean_rssi"] = df_test["ap_mean_rssi"]
    pred = tree.predict(X_test[FEATURES])
    dt_m = metrics(df_test[TARGET].to_numpy(), pred)
    print(f"  Decision Tree TEST: {dt_m}")

    # Full benchmark set required by the project brief. All candidates use the
    # same spatial test set and the same train-fold AP encoding. Only the small
    # Decision Tree is exported to firmware.
    candidate_rows = []
    if args.full:
        y_train = df_tv[TARGET].to_numpy()
        X_train = X_tv[FEATURES]
        X_test = X_test[FEATURES]
        candidates = {
            "linear_regression": LinearRegression(),
            "ridge_regression": Ridge(alpha=1.0),
            "random_forest": RandomForestRegressor(
                n_estimators=100, max_depth=16, min_samples_leaf=20,
                n_jobs=-1, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100, max_depth=3, min_samples_leaf=20,
                random_state=42),
        }
        for name, candidate in candidates.items():
            started = time.perf_counter()
            candidate.fit(X_train, y_train)
            if name == "ridge_regression":
                joblib.dump(candidate, TRAINED / "ridge_model.pkl")
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            candidate_pred = candidate.predict(X_test)
            predict_seconds = time.perf_counter() - started
            row = dict(model=name, **metrics(df_test[TARGET].to_numpy(), candidate_pred),
                       fit_seconds=fit_seconds,
                       inference_us_per_row=1e6 * predict_seconds / max(1, len(X_test)),
                       n_features=len(FEATURES))
            try:
                row["model_bytes"] = len(pickle.dumps(candidate, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception:
                row["model_bytes"] = np.nan
            candidate_rows.append(row)
            print(f"  {name}: {row}")

    # --- per-AP trees comparison ------------------------------------------
    print("Training per-AP Decision Trees (top measured APs) ...")
    pa = per_ap_trees(df_train, df_test, bmap, top_n=15)
    if len(pa):
        print(f"  per-AP trees mean test MAE = {pa['MAE'].mean():.3f} (over {len(pa)} APs)")

    # --- save model bundle ------------------------------------------------
    bundle = {
        "model_type": "DecisionTreeRegressor",
        "tree": tree,
        "feature_names": FEATURES,
        "building_map": bmap,
        "ap_mean_map": {k: float(v) for k, v in ap_mean_final.items()},
        "global_mean_rssi": float(gmean_final),
        "best_depth": best["depth"],
        "best_leaf": best["leaf"],
        "cv_mae": best["mae"],
        "dataset": args.dataset,
    }
    joblib.dump(bundle, TRAINED / "model_bundle.pkl")
    if args.dataset == "D5":
        joblib.dump(bundle, TRAINED / "benchmark_model_bundle.pkl")

    # --- metrics table ----------------------------------------------------
    rows = [dict(model=name, **mm) for name, mm in baseline_metrics.items()]
    rows.append(dict(model="decision_tree", **dt_m,
                     fit_seconds=np.nan, inference_us_per_row=np.nan,
                     n_features=len(FEATURES), model_bytes=len(pickle.dumps(tree))))
    rows.extend(candidate_rows)
    if len(pa):
        rows.append(dict(model="per_ap_tree_mean", MAE=float(pa["MAE"].mean()),
                          RMSE=float(pa["RMSE"].mean()), R2=float("nan")))
    mtab = pd.DataFrame(rows)
    mtab.to_csv(COMPARISON / "model_metrics.csv", index=False)
    print("wrote", COMPARISON / "model_metrics.csv")

    # --- feature analysis -------------------------------------------------
    imp = pd.DataFrame({"feature": FEATURES, "importance": tree.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(ANALYSIS / "feature_analysis.csv", index=False)
    print("wrote", ANALYSIS / "feature_analysis.csv")
    print("feature importances:\n", imp.to_string(index=False))

    # save test predictions for the evaluation script
    df_test = df_test.copy()
    df_test["pred_rssi"] = pred
    df_test[["building_id", "floor_id", "x", "y", "bssid", "rssi", "pred_rssi",
             "ap_mean_rssi", "cell_id"]].to_csv(COMPARISON / "test_predictions.csv", index=False)
    # save bundle meta as JSON (no large arrays) for the exporter
    meta = {k: v for k, v in bundle.items() if k not in ("tree", "ap_mean_map")}
    meta["n_ap_mean_entries"] = len(bundle["ap_mean_map"])
    (TRAINED / "feature_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    # save AP-mean map separately as JSON for the C++ exporter
    (TRAINED / "ap_mean_map.json").write_text(json.dumps(bundle["ap_mean_map"], indent=2), encoding="utf-8")
    print("wrote model_bundle.pkl, feature_meta.json, ap_mean_map.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
