"""COVERAGE-AI preprocessing (Stages 5-6): convert fingerprint matrices to the
common long format used by COVERAGE-AI.

Common schema (compatible with the CSV the ESP32 device will produce):

    timestamp, dataset_id, environment_id, building_id, floor_id, session_id,
    x, y, latitude, longitude, bssid, ssid, rssi, channel, frequency,
    encryption, measurement_id

Missing-value strategy (documented):
  * RSSI sentinel +100 means "access point NOT detected at this sample".
    Such cells are DROPPED (treated as missing), never read as +100 dBm.
  * Fields unavailable in a public dataset (ssid, channel, frequency,
    encryption, latitude/longitude when only local metres exist, timestamp,
    session_id) are left as empty (NaN), never invented.

Datasets are processed SEPARATELY (no blind merging) because they come from
different phones, antennas and environments -> domain shift. Outputs:
  datasets/processed/<dataset_id>_long.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / "datasets" / "extracted"
RAW = ROOT / "datasets" / "raw"
PROCESSED = ROOT / "datasets" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS = ROOT / "reports"

SENTINEL = 100  # +100 dBm means "not detected" for UJI / Tampere datasets
LONG_COLS = [
    "timestamp", "dataset_id", "environment_id", "building_id", "floor_id",
    "session_id", "x", "y", "latitude", "longitude", "bssid", "ssid", "rssi",
    "channel", "frequency", "encryption", "measurement_id",
]


def find_one(start, name):
    if start is None or not Path(start).exists():
        return None
    m = list(Path(start).rglob(name))
    return m[0] if m else None


def build_long(matrix, coords, dataset_id, building_label, ap_prefix,
               has_floor=True, has_z=True):
    """Vectorized melt of a (samples x APs) fingerprint matrix into long rows,
    keeping only DETECTED cells (rssi != SENTINEL)."""
    M = matrix if isinstance(matrix, pd.DataFrame) else pd.DataFrame(matrix)
    vals = M.to_numpy()
    mask = vals != SENTINEL
    sample_idx, ap_idx = np.where(mask)
    rssi = vals[mask]
    n = sample_idx.shape[0]

    def col(name):
        return coords[name].to_numpy()[sample_idx] if name in coords.columns else np.full(n, np.nan)

    x = col("x"); y = col("y"); z = col("z")
    floor = col("floor") if has_floor else np.full(n, np.nan)
    out = pd.DataFrame({
        "timestamp": np.full(n, np.nan, dtype=object),
        "dataset_id": dataset_id,
        "environment_id": "indoor",
        "building_id": building_label,
        "floor_id": floor,
        "session_id": np.full(n, np.nan, dtype=object),
        "x": x,
        "y": y,
        "latitude": np.full(n, np.nan),
        "longitude": np.full(n, np.nan),
        "bssid": [f"{ap_prefix}_{a}" for a in ap_idx],
        "ssid": np.full(n, np.nan, dtype=object),
        "rssi": rssi.astype(float),
        "channel": np.full(n, np.nan),
        "frequency": np.full(n, np.nan),
        "encryption": np.full(n, np.nan, dtype=object),
        "measurement_id": sample_idx,
    })
    # keep z as auxiliary (not in the official common schema but useful)
    if has_z and "z" in coords.columns:
        out["z"] = z
    return out


def process_d5():
    base = EXTRACTED / "D5"
    parts = []
    for b in ["TIE1", "SAH1"]:
        rp = find_one(base, f"{b}_training_rss.csv")
        cp = find_one(base, f"{b}_training_coordinates.csv")
        if rp is None or cp is None:
            print(f"  [D5/{b}] missing files, skip")
            continue
        M = pd.read_csv(rp, header=None)
        C = pd.read_csv(cp, header=None)
        nc = C.shape[1]
        if nc >= 4:
            C.columns = ["x", "y", "z", "floor"] + [f"c{i}" for i in range(4, nc)]
        elif nc == 3:
            C.columns = ["x", "y", "z"]
        else:
            C.columns = ["x", "y"] + [f"c{i}" for i in range(2, nc)]
        long = build_long(M, C, dataset_id="D5", building_label=b,
                          ap_prefix=f"D5_{b}", has_floor=("floor" in C.columns),
                          has_z=("z" in C.columns))
        print(f"  [D5/{b}] matrix {M.shape} -> long rows {len(long)} "
              f"(detected fraction {len(long)/(M.shape[0]*M.shape[1]):.3f})")
        parts.append(long)
    if parts:
        df = pd.concat(parts, ignore_index=True)
        out = PROCESSED / "D5_long.csv"
        df.to_csv(out, index=False)
        print(f"  wrote {out} ({len(df)} rows, {df.nunique()['bssid']} APs)")
        return df
    return None


def process_d1():
    base = EXTRACTED / "D1"
    rp = find_one(base, "trainingData.csv")
    if rp is None:
        print("  [D1] trainingData.csv missing, skip")
        return None
    df = pd.read_csv(rp)
    wap = [c for c in df.columns if str(c).upper().startswith("WAP")]
    M = df[wap]
    # UJI "LONGITUDE"/"LATITUDE" are local planar metres (not WGS84 degrees)
    C = pd.DataFrame({
        "x": df["LONGITUDE"].to_numpy(),
        "y": df["LATITUDE"].to_numpy(),
        "floor": df["FLOOR"].to_numpy(),
    })
    C["building_num"] = df["BUILDINGID"].to_numpy()
    M_vals = M.to_numpy()
    mask = M_vals != SENTINEL
    sample_idx, ap_idx = np.where(mask)
    rssi = M_vals[mask]
    n = sample_idx.shape[0]
    bld = df["BUILDINGID"].to_numpy()[sample_idx]
    out = pd.DataFrame({
        "timestamp": np.full(n, np.nan, dtype=object),
        "dataset_id": "D1",
        "environment_id": "indoor",
        "building_id": [f"BLD{int(b)}" for b in bld],
        "floor_id": C["floor"].to_numpy()[sample_idx].astype(float),
        "session_id": np.full(n, np.nan, dtype=object),
        "x": C["x"].to_numpy()[sample_idx],
        "y": C["y"].to_numpy()[sample_idx],
        "latitude": np.full(n, np.nan),
        "longitude": np.full(n, np.nan),
        "bssid": [f"D1_AP{a}" for a in ap_idx],
        "ssid": np.full(n, np.nan, dtype=object),
        "rssi": rssi.astype(float),
        "channel": np.full(n, np.nan),
        "frequency": np.full(n, np.nan),
        "encryption": np.full(n, np.nan, dtype=object),
        "measurement_id": sample_idx,
    })
    o = PROCESSED / "D1_long.csv"
    out.to_csv(o, index=False)
    print(f"  [D1] matrix {M.shape} -> long rows {len(out)} "
          f"(detected fraction {len(out)/(M.shape[0]*M.shape[1]):.3f}); wrote {o}")
    return out


def process_d4():
    base = RAW / "zenodo" / "tampere_wlan_rss_1161525"
    rp = find_one(base, "Training_rss.csv")
    cp = find_one(base, "Training_coordinates.csv")
    if rp is None or cp is None:
        print("  [D4] missing files, skip")
        return None
    M = pd.read_csv(rp, header=None)
    C = pd.read_csv(cp, header=None)
    nc = C.shape[1]
    C.columns = ["x", "y", "z"][:nc] + [f"c{i}" for i in range(nc, nc)]
    long = build_long(M, C, dataset_id="D4", building_label="TAMP_WLAN",
                      ap_prefix="D4", has_floor=False, has_z=("z" in C.columns))
    o = PROCESSED / "D4_long.csv"
    long.to_csv(o, index=False)
    print(f"  [D4] matrix {M.shape} -> long rows {len(long)} "
          f"(detected fraction {len(long)/(M.shape[0]*M.shape[1]):.3f}); wrote {o}")
    return long


def main():
    print("Processing D5 (primary) ...")
    d5 = process_d5()
    print("Processing D1 (secondary) ...")
    d1 = process_d1()
    print("Processing D4 (secondary) ...")
    d4 = process_d4()

    # short preprocessing report
    lines = ["# COVERAGE-AI — Preprocessing Report (Stages 5-6)\n",
             "## Missing-value strategy\n",
             "- RSSI sentinel **+100 dBm** means *access point not detected*. ",
             "Such cells are dropped (treated as missing), never read as +100 dBm.\n",
             "- Fields unavailable in a public dataset (ssid, channel, frequency, ",
             "encryption, latitude/longitude, timestamp, session_id) are left empty.\n",
             "- Datasets are processed SEPARATELY (no blind merging) due to domain shift ",
             "(different phones/antennas/environments).\n\n",
             "## Common schema\n",
             "Columns: `" + "`, `".join(LONG_COLS) + "` (matches the CSV the ESP32 produces).\n\n",
             "## Output files (datasets/processed/)\n"]
    for name, df in [("D5_long.csv", d5), ("D1_long.csv", d1), ("D4_long.csv", d4)]:
        if df is None:
            lines.append(f"- **{name}**: not produced\n")
            continue
        lines.append(f"- **{name}**: {len(df)} rows; "
                     f"{df['bssid'].nunique()} APs; buildings={sorted(df['building_id'].unique().tolist())}; "
                     f"RSSI range [{df['rssi'].min():.1f}, {df['rssi'].max():.1f}] dBm, "
                     f"median {df['rssi'].median():.1f} dBm\n")
    lines.append("\n## Domain-shift caveat\n")
    lines.append("Public-dataset RSSI was recorded with smartphones; ESP32 RSSI will differ. ")
    lines.append("Models trained here are benchmarks only and must be fine-tuned on ESP32 data.\n")
    (REPORTS / "preprocessing_report.md").write_text("".join(lines), encoding="utf-8")
    print("wrote", REPORTS / "preprocessing_report.md")


if __name__ == "__main__":
    main()
