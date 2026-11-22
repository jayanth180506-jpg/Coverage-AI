"""COVERAGE-AI dataset inspection (Stage 4).

Inspects every downloaded dataset regardless of layout:
  * UJIIndoorLoc (D1): wide WAP columns + inline LONGITUDE/LATITUDE/FLOOR/...
  * Tampere WLAN RSS (D4): separate RSS matrix + coordinates files.
  * Tampere multi-building (D5): per-building RSS matrix + coordinates files.
  * Crowdsourced (D6): FINGERPRINTING_DB RSS + coordinates.

Auto-detects the "not detected" sentinel (commonly +100 dBm), reports shape,
dtypes, missing values, duplicates, invalid RSSI, coordinate ranges, AP ids,
indoor/outdoor assumption, and writes:
  * analysis/dataset_quality_report.csv
  * analysis/figures/*.png
  * reports/inspection_report.md
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
EXTRACTED = ROOT / "datasets" / "extracted"
ANALYSIS = ROOT / "analysis"
FIG = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
FIG.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# RSSI physics: real Wi-Fi RSSI is in roughly [-110, -10] dBm. Any value far
# outside that range (e.g. +100, 0) is a sentinel/invalid, not a real RSSI.
RSSI_VALID_MIN = -120.0
RSSI_VALID_MAX = -10.0
COMMON_SENTINELS = (100, 110, 0)


# --- discovery config ------------------------------------------------------
def find_one(start: Path, name: str):
    """Recursively find a file by exact name under `start` (handles nested zip
    extraction folders). Returns the first match or None."""
    if start is None or not start.exists():
        return None
    matches = list(start.rglob(name))
    return matches[0] if matches else None


def d1_files():
    base = EXTRACTED / "D1"
    out = []
    for tag, desc, fn in [("D1_train", "UJIIndoorLoc training", "trainingData.csv"),
                          ("D1_val", "UJIIndoorLoc validation", "validationData.csv")]:
        p = find_one(base, fn)
        if p:
            out.append((tag, desc, "inline", p))
    return out


def d4_files():
    base = RAW / "zenodo" / "tampere_wlan_rss_1161525"
    out = []
    for tag, desc, rf, cf in [
        ("D4_train", "Tampere WLAN RSS training", "Training_rss.csv", "Training_coordinates.csv"),
        ("D4_test", "Tampere WLAN RSS test", "Test_rss.csv", "Test_coordinates.csv"),
    ]:
        rp, cp = find_one(base, rf), find_one(base, cf)
        if rp and cp:
            out.append((tag, desc, "matrix", rp, cp))
    return out


def d5_files():
    base = EXTRACTED / "D5"
    out = []
    for b in ["TIE1", "SAH1"]:
        for tag_s, desc_s, rf, cf in [
            ("train", "training", f"{b}_training_rss.csv", f"{b}_training_coordinates.csv"),
            ("test", "test", f"{b}_test_rss.csv", f"{b}_test_coordinates.csv"),
        ]:
            rp, cp = find_one(base, rf), find_one(base, cf)
            if rp and cp:
                out.append((f"D5_{b}_{tag_s}", f"Tampere {b} {desc_s}", "matrix", rp, cp))
    return out


def d6_files():
    base = EXTRACTED / "D6"
    out = []
    for tag, desc, rf, cf in [
        ("D6_train", "Crowdsourced training", "Training_rss_21Aug17.csv", "Training_coordinates_21Aug17.csv"),
        ("D6_test", "Crowdsourced test", "Test_rss_21Aug17.csv", "Test_coordinates_21Aug17.csv"),
    ]:
        rp, cp = find_one(base, rf), find_one(base, cf)
        if rp and cp:
            out.append((tag, desc, "matrix", rp, cp))
    return out


# --- helpers ---------------------------------------------------------------
def detect_sentinel(matrix_df):
    """Return (sentinel_value, sentinel_fraction) or (None, 0.0)."""
    vals = matrix_df.to_numpy().ravel()
    if vals.size == 0:
        return None, 0.0
    # candidate: the most frequent value if it is positive and outside dBm range
    try:
        counts = pd.Series(vals).value_counts(dropna=False)
    except Exception:
        counts = pd.Series(vals).value_counts()
    top_val, top_cnt = next(iter(counts.items()))
    top_frac = top_cnt / vals.size
    # explicit common sentinels first
    for s in COMMON_SENTINELS:
        c = counts.get(s, 0)
        if c / vals.size > 0.05:
            return float(s), float(c / vals.size)
    if top_frac > 0.05 and pd.notna(top_val) and float(top_val) > RSSI_VALID_MAX:
        return float(top_val), float(top_frac)
    return None, float(counts.get(np.nan, 0) / vals.size)


def real_rssi_stats(matrix_df, sentinel):
    vals = matrix_df.to_numpy().ravel()
    mask = np.ones(vals.shape, dtype=bool)
    if sentinel is not None:
        mask &= vals != sentinel
    mask &= np.isfinite(vals) if np.issubdtype(vals.dtype, np.floating) else np.ones_like(mask)
    # keep only physically plausible dBm
    rv = vals[mask & (vals >= RSSI_VALID_MIN) & (vals <= RSSI_VALID_MAX)]
    if rv.size == 0:
        return dict(n=0)
    return dict(n=int(rv.size), min=float(rv.min()), max=float(rv.max()),
                mean=float(rv.mean()), median=float(np.median(rv)),
                p10=float(np.percentile(rv, 10)), p90=float(np.percentile(rv, 90)))


def coord_cols(df):
    low = {c.lower(): c for c in df.columns}
    out = {}
    for key, names in [("x", ["x", "x_m", "xcoord"]),
                      ("y", ["y", "y_m", "ycoord"]),
                      ("z", ["z", "z_m", "zcoord", "height"]),
                      ("floor", ["floor", "floorid", "floor_id"]),
                      ("lat", ["latitude", "lat"]),
                      ("lon", ["longitude", "lon", "long", "lng"]),
                      ("building", ["buildingid", "building_id", "building"])]:
        for n in names:
            if n in low:
                out[key] = low[n]
                break
    return out


def inspect_inline_uji(fpath):
    """D1: WAP columns + inline location columns."""
    df = pd.read_csv(fpath)
    wap = [c for c in df.columns if str(c).upper().startswith("WAP")]
    meta = [c for c in df.columns if not str(c).upper().startswith("WAP")]
    matrix = df[wap]
    sentinel, sfrac = detect_sentinel(matrix)
    stats = real_rssi_stats(matrix, sentinel)
    cc = coord_cols(df)
    coord_ranges = {}
    for k in ["lon", "lat", "floor", "building"]:
        if k in cc:
            s = df[cc[k]]
            coord_ranges[k] = (float(s.min()), float(s.max()), int(s.nunique()))
    ndup = int(df.duplicated().sum())
    n_samples, n_aps = df.shape[0], len(wap)
    return dict(
        file=str(fpath.name), role="inline_wide", n_samples=n_samples, n_aps=n_aps,
        dtypes=str({c: str(df[c].dtype) for c in meta}),
        sentinel=sentinel, sentinel_fraction=sfrac, rssi_stats=stats,
        coord_ranges=coord_ranges, n_duplicates=ndup, meta_cols=meta,
        df=df, matrix=matrix,
    )


def inspect_matrix(fpath, cpath):
    # RSS and coordinate files for Tampere / crowdsourced datasets have NO
    # header row; read positionally so the first data row is not consumed.
    matrix = pd.read_csv(fpath, header=None)
    matrix.columns = [f"AP{i}" for i in range(matrix.shape[1])]
    coords = pd.read_csv(cpath, header=None)
    nc = coords.shape[1]
    if nc >= 4:
        coords.columns = ["x", "y", "z", "floor"] + [f"c{i}" for i in range(4, nc)]
    elif nc == 3:
        coords.columns = ["x", "y", "z"]
    elif nc == 2:
        coords.columns = ["x", "y"]
    else:
        coords.columns = [f"c{i}" for i in range(nc)]
    sentinel, sfrac = detect_sentinel(matrix)
    stats = real_rssi_stats(matrix, sentinel)
    coord_ranges = {}
    for k in ["x", "y", "z", "floor"]:
        if k in coords.columns:
            s = coords[k]
            coord_ranges[k] = (float(s.min()), float(s.max()), int(s.nunique()))
    ndup = int(matrix.duplicated().sum())
    n_samples, n_aps = matrix.shape[0], matrix.shape[1]
    return dict(
        file=str(fpath.name), role="matrix+coords", n_samples=n_samples, n_aps=n_aps,
        dtypes=str(matrix.dtypes.iloc[0]),
        sentinel=sentinel, sentinel_fraction=sfrac, rssi_stats=stats,
        coord_ranges=coord_ranges, n_duplicates=ndup,
        coords_cols=list(coords.columns), matrix=matrix, coords=coords,
    )


# --- plotting --------------------------------------------------------------
def plot_rssi_dist(stats_holder, tag):
    # collect all real rssi from the matrix
    matrix = stats_holder["matrix"]
    sentinel = stats_holder.get("sentinel")
    vals = matrix.to_numpy().ravel()
    mask = np.ones(vals.shape, dtype=bool)
    if sentinel is not None:
        mask &= vals != sentinel
    rv = vals[mask & (vals >= RSSI_VALID_MIN) & (vals <= RSSI_VALID_MAX)]
    if rv.size == 0:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(rv, bins=60, color="#2a6fdb")
    plt.xlabel("RSSI (dBm)")
    plt.ylabel("count")
    plt.title(f"{tag}: detected-RSSI distribution")
    plt.tight_layout()
    plt.savefig(FIG / f"{tag}_rssi_dist.png", dpi=110)
    plt.close()


def plot_ap_detection(matrix, sentinel, tag):
    if sentinel is not None:
        det = (matrix != sentinel).sum(axis=0)
    else:
        det = matrix.notna().sum(axis=0)
    plt.figure(figsize=(8, 4))
    plt.hist(det, bins=50, color="#1f9d55")
    plt.xlabel("number of samples that detected the AP")
    plt.ylabel("number of APs")
    plt.title(f"{tag}: detections per AP")
    plt.tight_layout()
    plt.savefig(FIG / f"{tag}_ap_detections.png", dpi=110)
    plt.close()


def plot_spatial(coords, xcol, ycol, tag, c=None):
    plt.figure(figsize=(6, 6))
    n = len(coords)
    idx = np.random.RandomState(42).choice(n, size=min(n, 5000), replace=False)
    xs = coords[xcol].iloc[idx]
    ys = coords[ycol].iloc[idx]
    if c is not None:
        cv = np.asarray(c)
        order = np.argsort(cv[idx])
        plt.scatter(xs.iloc[order], ys.iloc[order], c=cv[idx][order],
                    s=6, cmap="viridis", alpha=0.7)
        plt.colorbar(label="mean RSSI (dBm)")
    else:
        plt.scatter(xs, ys, s=6, alpha=0.5, color="#444")
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(f"{tag}: spatial distribution of samples")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(FIG / f"{tag}_spatial.png", dpi=110)
    plt.close()


def plot_missing_per_ap(matrix, sentinel, tag):
    if sentinel is not None:
        miss = (matrix == sentinel).mean(axis=0)
    else:
        miss = matrix.isna().mean(axis=0)
    plt.figure(figsize=(8, 4))
    plt.hist(miss, bins=50, color="#e07b39")
    plt.xlabel("fraction of samples where AP not detected (sentinel)")
    plt.ylabel("number of APs")
    plt.title(f"{tag}: not-detected fraction per AP")
    plt.tight_layout()
    plt.savefig(FIG / f"{tag}_missing_per_ap.png", dpi=110)
    plt.close()


# --- main ------------------------------------------------------------------
def summarize_row(tag, desc, r):
    rs = r.get("rssi_stats", {})
    coord = r.get("coord_ranges", {})
    def cr(k):
        v = coord.get(k)
        return f"{v[0]:.3f}..{v[1]:.3f} ({v[2]} uniq)" if v else "NA"
    return {
        "tag": tag, "description": desc, "file": r.get("file", ""),
        "role": r.get("role", ""), "n_samples": r.get("n_samples", ""),
        "n_aps": r.get("n_aps", ""), "sentinel": r.get("sentinel", ""),
        "sentinel_fraction": round(r.get("sentinel_fraction", 0) or 0, 4),
        "rssi_n": rs.get("n", ""), "rssi_min": rs.get("min", ""),
        "rssi_max": rs.get("max", ""), "rssi_mean": rs.get("mean", ""),
        "rssi_median": rs.get("median", ""),
        "lon_or_x": cr("lon") if "lon" in coord else cr("x"),
        "lat_or_y": cr("lat") if "lat" in coord else cr("y"),
        "z": cr("z"), "floor": cr("floor"), "building": cr("building"),
        "n_duplicates": r.get("n_duplicates", ""),
        "coords_cols": ",".join(r.get("coords_cols", [])) if r.get("coords_cols") else "",
        "meta_cols": ",".join(r.get("meta_cols", [])) if r.get("meta_cols") else "",
        "indoor_outdoor": "indoor (campus/university)",
    }


def main():
    rows = []
    md = ["# COVERAGE-AI — Dataset Inspection Report (Stage 4)\n"]

    # D1 (inline wide)
    for tag, desc, _, fpath in d1_files():
        if not fpath.exists():
            md.append(f"- **{tag}**: file missing ({fpath.name})\n")
            continue
        r = inspect_inline_uji(fpath)
        rows.append(summarize_row(tag, desc, r))
        plot_rssi_dist(r, tag)
        plot_ap_detection(r["matrix"], r.get("sentinel"), tag)
        plot_missing_per_ap(r["matrix"], r.get("sentinel"), tag)
        cc = coord_cols(r["df"])
        if "lon" in cc and "lat" in cc:
            plot_spatial(r["df"], cc["lon"], cc["lat"], tag)
        md.append(f"- **{tag}** ({desc}): {r['n_samples']} samples × {r['n_aps']} APs; "
                  f"sentinel={r.get('sentinel')} (frac={r.get('sentinel_fraction'):.3f}); "
                  f"RSSI mean={r['rssi_stats'].get('mean','NA')}, median={r['rssi_stats'].get('median','NA')}\n")

    # D4, D5, D6 (matrix + coords)
    for tag, desc, kind, fpath, cpath in (d4_files() + d5_files() + d6_files()):
        if not fpath.exists() or not cpath.exists():
            md.append(f"- **{tag}**: file missing (rss={fpath.exists()}, coords={cpath.exists()})\n")
            continue
        r = inspect_matrix(fpath, cpath)
        rows.append(summarize_row(tag, desc, r))
        plot_rssi_dist(r, tag)
        plot_ap_detection(r["matrix"], r.get("sentinel"), tag)
        plot_missing_per_ap(r["matrix"], r.get("sentinel"), tag)
        if "x" in r["coords"].columns and "y" in r["coords"].columns:
            # mean detected RSSI per sample for color
            mat = r["matrix"]; sent = r.get("sentinel")
            m = mat.where(mat != sent, np.nan) if sent is not None else mat
            sample_mean = m.mean(axis=1, skipna=True).to_numpy()
            plot_spatial(r["coords"], "x", "y", tag, c=sample_mean)
        md.append(f"- **{tag}** ({desc}): {r['n_samples']} samples × {r['n_aps']} APs; "
                  f"sentinel={r.get('sentinel')} (frac={r.get('sentinel_fraction'):.3f}); "
                  f"coords cols={r.get('coords_cols')}; "
                  f"RSSI mean={r['rssi_stats'].get('mean','NA')}, median={r['rssi_stats'].get('median','NA')}\n")

    # write quality report CSV
    import csv
    qpath = ANALYSIS / "dataset_quality_report.csv"
    cols = ["tag", "description", "file", "role", "n_samples", "n_aps", "sentinel",
            "sentinel_fraction", "rssi_n", "rssi_min", "rssi_max", "rssi_mean",
            "rssi_median", "lon_or_x", "lat_or_y", "z", "floor", "building",
            "n_duplicates", "coords_cols", "meta_cols", "indoor_outdoor"]
    with open(qpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})
    print("wrote", qpath, "rows=", len(rows))

    # markdown
    md.append("\n## Suitability for RSSI prediction\n")
    md.append("- D5 (Tampere multi-building): metre coordinates + documented +100 sentinel + 2 buildings -> **PRIMARY**.\n")
    md.append("- D1 (UJIIndoorLoc): large, documented +100, geographic coords (need conversion) -> secondary.\n")
    md.append("- D4 (Tampere WLAN RSS): smaller single-building fingerprint -> secondary.\n")
    md.append("- D6 (Crowdsourced): heterogeneous crowdsourcing -> localization/propagation reference only.\n")
    md.append("\nFigures are in `analysis/figures/`.\n")
    (REPORTS / "inspection_report.md").write_text("".join(md), encoding="utf-8")
    print("wrote", REPORTS / "inspection_report.md")
    print("figures in", FIG)


if __name__ == "__main__":
    main()
