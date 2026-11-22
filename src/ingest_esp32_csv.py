"""Normalize a device-export CSV into the common training schema."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ["timestamp", "dataset_id", "environment_id", "building_id", "floor_id",
          "session_id", "x", "y", "latitude", "longitude", "bssid", "ssid",
          "rssi", "channel", "frequency", "encryption", "measurement_id"]
ALIASES = {"time": "timestamp", "t": "timestamp", "building": "building_id",
           "floor": "floor_id", "ap": "bssid", "mac": "bssid", "signal": "rssi",
           "rssi_dbm": "rssi", "seq": "measurement_id", "x_m": "x",
           "y_m": "y", "gps_lat": "latitude", "gps_lon": "longitude",
           "channel_id": "channel"}


def normalize(path, output):
    d = pd.read_csv(path)
    d.columns = [ALIASES.get(str(c).strip().lower(), str(c).strip().lower()) for c in d.columns]
    required = {"bssid", "rssi", "x", "y"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"CSV lacks required fields: {', '.join(sorted(missing))}")
    if "measurement_id" not in d:
        d["measurement_id"] = np.arange(len(d), dtype=np.int64)
    if "floor_id" not in d:
        d["floor_id"] = 0
    if "building_id" not in d:
        d["building_id"] = "SITE"
    for col in SCHEMA:
        if col not in d:
            d[col] = np.nan
    d["dataset_id"] = "ESP32"
    d["environment_id"] = d["environment_id"].fillna("unknown")
    d["bssid"] = d["bssid"].astype(str).str.strip().str.lower()
    d["rssi"] = pd.to_numeric(d["rssi"], errors="coerce")
    d["x"] = pd.to_numeric(d["x"], errors="coerce")
    d["y"] = pd.to_numeric(d["y"], errors="coerce")
    d["floor_id"] = pd.to_numeric(d["floor_id"], errors="coerce").fillna(0).astype(int)
    invalid = (d["bssid"].isin(["", "nan"]) | d["rssi"].isna() |
               d["x"].isna() | d["y"].isna() | (d["rssi"] < -120) | (d["rssi"] > 0))
    rejected = d[invalid].copy()
    accepted = d[~invalid][SCHEMA].copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    accepted.to_csv(output, index=False)
    if len(rejected):
        rejected.to_csv(output.with_name(output.stem + "_rejected.csv"), index=False)
    print(f"accepted={len(accepted)} rejected={len(rejected)} output={output}")
    return accepted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="ESP32-exported CSV")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "datasets" / "processed" / "ESP32_long.csv")
    args = ap.parse_args()
    normalize(args.csv, args.output)


if __name__ == "__main__":
    main()
