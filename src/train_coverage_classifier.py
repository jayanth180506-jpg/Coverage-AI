"""Separate RSSI-to-coverage classification objective with spatial holdout."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["very_poor", "poor", "fair", "good", "excellent"]


def classify(rssi):
    return np.select([rssi < -80, rssi < -70, rssi < -60, rssi < -50],
                     LABELS[:4], default="excellent")


def cells(df, bins=5):
    out = df.copy()
    for _, g in out.groupby("building_id"):
        x = np.linspace(g.x.min()-1e-8, g.x.max()+1e-8, bins+1)
        y = np.linspace(g.y.min()-1e-8, g.y.max()+1e-8, bins+1)
        out.loc[g.index, "cx"] = np.digitize(g.x, x[1:-1])
        out.loc[g.index, "cy"] = np.digitize(g.y, y[1:-1])
    out["cell"] = out.building_id.astype(str)+"_"+out.cx.astype(int).astype(str)+"_"+out.cy.astype(int).astype(str)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="D5")
    ap.add_argument("--thresholds", type=Path, help="JSON overrides: excellent_min, good_min, fair_min, poor_min")
    args = ap.parse_args()
    d = pd.read_csv(ROOT / "datasets" / "processed" / f"{args.dataset}_long.csv")
    thresholds = {"excellent_min": -50, "good_min": -60, "fair_min": -70, "poor_min": -80}
    if args.thresholds:
        thresholds.update(json.loads(args.thresholds.read_text(encoding="utf-8")))
    d = cells(d)
    y = np.select([d.rssi < thresholds["poor_min"], d.rssi < thresholds["fair_min"],
                   d.rssi < thresholds["good_min"], d.rssi < thresholds["excellent_min"]],
                  LABELS[:4], default="excellent")
    groups = d.cell.to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(d, y, groups))
    features = ["x", "y", "floor_id"]
    X = d[features].copy()
    X["building_num"] = d.building_id.astype("category").cat.codes
    # AP identity is represented as a training-only average signal feature.
    apmean = d.iloc[tr].groupby("bssid").rssi.mean()
    X["ap_mean_rssi"] = d.bssid.map(apmean).fillna(d.iloc[tr].rssi.mean())
    X = X[["x", "y", "floor_id", "building_num", "ap_mean_rssi"]]
    candidates = {"majority_baseline": DummyClassifier(strategy="most_frequent"),
                  "decision_tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=30, random_state=42),
                  "random_forest": RandomForestClassifier(n_estimators=80, max_depth=14,
                                                            min_samples_leaf=10, n_jobs=-1, random_state=42)}
    rows = []
    for name, model in candidates.items():
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[te])
        rows.append({"model": name, "accuracy": accuracy_score(y[te], pred),
                     "macro_f1": f1_score(y[te], pred, average="macro", zero_division=0)})
        if name == "random_forest":
            report = classification_report(y[te], pred, labels=LABELS, output_dict=True, zero_division=0)
            (ROOT / "models" / "comparison" / "coverage_classification_report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
            pd.DataFrame(confusion_matrix(y[te], pred, labels=LABELS), index=LABELS,
                         columns=LABELS).to_csv(ROOT / "models" / "comparison" / "coverage_confusion_matrix.csv")
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "models" / "comparison" / "coverage_classification_metrics.csv", index=False)
    lines = ["# Coverage Classification Evaluation\n\n",
             "Classes derive from measured RSSI using configurable engineering thresholds. ",
             "Spatial cells are held out; classification is a separate task from RSSI regression.\n\n",
             f"Thresholds (dBm): excellent >= {thresholds['excellent_min']}; good >= {thresholds['good_min']}; ",
             f"fair >= {thresholds['fair_min']}; poor >= {thresholds['poor_min']}; very poor < {thresholds['poor_min']}.\n\n",
             out.to_markdown(index=False), "\n\n",
             "Class imbalance and public-to-ESP32 domain shift limit interpretation. Firmware can use direct configurable thresholds on measured or predicted RSSI.\n"]
    (ROOT / "reports" / "coverage_classification.md").write_text("".join(lines), encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
