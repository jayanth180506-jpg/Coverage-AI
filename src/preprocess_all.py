"""Build separate common-format outputs for every downloaded fingerprint split.

The source's +100 sentinel is missing AP data, never an RSSI measurement.
Official/source splits remain in separate files; they are never concatenated for
training. Coordinates and identities are preserved or anonymized by AP column.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "datasets" / "extracted"
RAW = ROOT / "datasets" / "raw"
OUT = ROOT / "datasets" / "processed"
SENTINEL = 100


def coords_file(path):
    c = pd.read_csv(path, header=None)
    names = ["x", "y", "z", "floor"][:c.shape[1]]
    names += [f"c{i}" for i in range(len(names), c.shape[1])]
    c.columns = names
    return c


def matrix_long(matrix, coords, dataset, building, ap_prefix, dates=None,
                sessions=None):
    if len(matrix) != len(coords):
        raise ValueError(f"{dataset}/{building}: RSS rows {len(matrix)} != coordinate rows {len(coords)}")
    if dates is not None and len(dates) != len(matrix):
        raise ValueError(f"{dataset}/{building}: date row count mismatch")
    if sessions is not None and len(sessions) != len(matrix):
        raise ValueError(f"{dataset}/{building}: device row count mismatch")
    a = matrix.to_numpy()
    si, ai = np.where(a != SENTINEL)
    n = len(si)
    def coordinate(col):
        return coords[col].to_numpy()[si] if col in coords else np.full(n, np.nan)
    return pd.DataFrame({
        "timestamp": np.asarray(dates, dtype=object)[si] if dates is not None else np.nan,
        "dataset_id": dataset, "environment_id": "indoor", "building_id": building,
        "floor_id": coordinate("floor"),
        "session_id": np.asarray(sessions, dtype=object)[si] if sessions is not None else np.nan,
        "x": coordinate("x"), "y": coordinate("y"), "latitude": np.nan,
        "longitude": np.nan, "bssid": [f"{ap_prefix}_{i:03d}" for i in ai],
        "ssid": np.nan, "rssi": a[si, ai].astype(float), "channel": np.nan,
        "frequency": np.nan, "encryption": np.nan, "measurement_id": si,
        "z": coordinate("z"),
    })


def write_split(dataset, split, frames):
    if not frames:
        return None
    data = pd.concat(frames, ignore_index=True)
    path = OUT / f"{dataset}_{split}_long.csv"
    data.to_csv(path, index=False)
    if split == "training" and dataset in {"D4", "D5"}:
        data.to_csv(OUT / f"{dataset}_long.csv", index=False)
    print(f"{path.relative_to(ROOT)}: {len(data)} detected AP readings")
    return data


def process_d1():
    base = EXT / "D1" / "UJIndoorLoc"
    all_data = {}
    for split, filename in [("training", "trainingData.csv"), ("validation", "validationData.csv")]:
        path = base / filename
        if not path.exists():
            print(f"D1 {split}: missing {path}")
            continue
        d = pd.read_csv(path)
        wap = [col for col in d if str(col).upper().startswith("WAP")]
        coords = pd.DataFrame({"x": d["LONGITUDE"], "y": d["LATITUDE"], "floor": d["FLOOR"]})
        result = matrix_long(d[wap], coords, "D1", "", "D1_AP")
        # Expand building identity via measurement index.
        rowidx = result["measurement_id"].to_numpy(dtype=int)
        apidx = result["bssid"].str.rsplit("_", n=1).str[-1].to_numpy()
        result["building_id"] = [f"BLD{int(v)}" for v in d["BUILDINGID"].to_numpy()[rowidx]]
        result["bssid"] = "D1_AP" + apidx
        all_data[split] = result
        result.to_csv(OUT / f"D1_{split}_long.csv", index=False)
        print(f"datasets/processed/D1_{split}_long.csv: {len(result)} rows")
    if "training" in all_data:
        all_data["training"].to_csv(OUT / "D1_long.csv", index=False)


def process_matrix_dataset(dataset, base, buildings, splits):
    for split, source_label in splits:
        frames = []
        for building in buildings:
            prefix = f"{building}_" if building else ""
            rss = next(iter(base.rglob(f"{prefix}{source_label}_rss.csv")), None)
            coord = next(iter(base.rglob(f"{prefix}{source_label}_coordinates.csv")), None)
            # D4 filenames use Training/Test capitalization; match case-insensitively.
            if rss is None:
                rss = next((p for p in base.rglob("*.csv") if p.name.lower() == f"{prefix}{source_label}_rss.csv".lower()), None)
            if coord is None:
                coord = next((p for p in base.rglob("*.csv") if p.name.lower() == f"{prefix}{source_label}_coordinates.csv".lower()), None)
            if not rss or not coord:
                continue
            m = pd.read_csv(rss, header=None)
            c = coords_file(coord)
            frames.append(matrix_long(m, c, dataset, building or dataset,
                                      f"{dataset}_{building}" if building else dataset))
        write_split(dataset, split, frames)


def process_d6():
    base = EXT / "D6" / "FINGERPRINTING_DB"
    for label in ["Training", "Test"]:
        files = {kind: next(iter(base.glob(f"{label}_{kind}_21Aug17.csv")), None)
                 for kind in ["rss", "coordinates", "date", "device"]}
        if any(v is None for v in files.values()):
            print(f"D6 {label}: incomplete synchronized source files")
            continue
        m = pd.read_csv(files["rss"], header=None)
        c = coords_file(files["coordinates"])
        dates = pd.read_csv(files["date"], header=None).iloc[:, 0].astype(str).to_numpy()
        devices = pd.read_csv(files["device"], header=None).iloc[:, 0].astype(str).to_numpy()
        frame = matrix_long(m, c, "D6", "CROWDSOURCED", "D6", dates, devices)
        frame.to_csv(OUT / f"D6_{label.lower()}_long.csv", index=False)
        print(f"datasets/processed/D6_{label.lower()}_long.csv: {len(frame)} rows")
    print("D6 source README conflicts on train/test semantics; filenames retained verbatim.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    process_d1()
    process_matrix_dataset("D4", RAW / "zenodo" / "tampere_wlan_rss_1161525",
                           [""], [("training", "Training"), ("test", "Test")])
    process_matrix_dataset("D5", EXT / "D5", ["TIE1", "SAH1"],
                           [("training", "training"), ("test", "test")])
    process_d6()
    entries = []
    for p in sorted(OUT.glob("*_long.csv")):
        df = pd.read_csv(p, usecols=["dataset_id", "building_id", "bssid", "rssi"])
        entries.append(f"- `{p.name}`: {len(df)} detected readings; {df.bssid.nunique()} AP identifiers; RSSI {df.rssi.min():.1f} to {df.rssi.max():.1f} dBm.\n")
    text = ["# Preprocessing Report\n\n",
            "The +100 dBm source sentinel is dropped as undetected AP data. Unavailable common-schema fields remain missing. Source train/validation/test files remain separate; D5_long.csv and D4_long.csv are convenience training views only.\n\n",
            "## Generated files\n", *entries,
            "\nD6 source split names are preserved because its bundled README conflicts on their train/test semantics. D1 coordinates are projected planar values; exact CRS/origin remains unknown. Public-device RSSI requires ESP32-specific calibration.\n"]
    (ROOT / "reports" / "preprocessing_report.md").write_text("".join(text), encoding="utf-8")


if __name__ == "__main__":
    main()
