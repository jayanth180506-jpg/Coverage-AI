"""Summarize measured and predicted weak-signal grid cells from spatial test data."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=-70,
                    help="weak-signal threshold in dBm (default -70)")
    args = ap.parse_args()
    p = ROOT / "models" / "comparison" / "test_predictions.csv"
    d = pd.read_csv(p)
    region = []
    for building, g in d.groupby("building_id"):
        xedges = np.linspace(g.x.min()-1e-8, g.x.max()+1e-8, 6)
        yedges = np.linspace(g.y.min()-1e-8, g.y.max()+1e-8, 6)
        xbin = np.digitize(g.x, xedges[1:-1])
        ybin = np.digitize(g.y, yedges[1:-1])
        tmp = g.assign(xbin=xbin, ybin=ybin)
        for (xb, yb), cell in tmp.groupby(["xbin", "ybin"]):
            region.append({"building_id": building, "xbin": int(xb), "ybin": int(yb),
                           "n_readings": len(cell), "mean_measured_rssi": cell.rssi.mean(),
                           "mean_predicted_rssi": cell.pred_rssi.mean(),
                           "measured_weak_fraction": float((cell.rssi < args.threshold).mean()),
                           "predicted_weak_fraction": float((cell.pred_rssi < args.threshold).mean()),
                           "predicted_weak": bool(cell.pred_rssi.mean() < args.threshold)})
    out = pd.DataFrame(region).sort_values(["building_id", "ybin", "xbin"])
    out.to_csv(ROOT / "models" / "comparison" / "weak_coverage_regions.csv", index=False)
    thresholded = out[out.predicted_weak]
    report = ["# Weak Coverage Regions\n\n",
              f"Threshold: predicted/measured RSSI below {args.threshold:g} dBm. ",
              "Summaries use held-out spatial test predictions, grouped into 5x5 bins per building. ",
              "Binning is a coarse report, not a router-placement recommendation.\n\n",
              f"- Regions summarized: {len(out)}\n",
              f"- Regions with mean predicted RSSI below threshold: {len(thresholded)}\n",
              "- Detailed measured and predicted weak fractions: `models/comparison/weak_coverage_regions.csv`\n"]
    (ROOT / "reports" / "weak_coverage.md").write_text("".join(report), encoding="utf-8")
    print(f"regions={len(out)} predicted_weak={len(thresholded)} threshold={args.threshold} dBm")


if __name__ == "__main__":
    main()
