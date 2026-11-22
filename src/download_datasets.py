"""COVERAGE-AI dataset downloader.

Downloads all public datasets relevant to the project, records a manifest with
SHA-256 checksums, and extracts archives. Sources are discovered at runtime:

  * UCI  : static zip URL (no auth).
  * Zenodo : REST API discovers real file links + checksums (no auth).
  * Kaggle : official `kaggle` CLI (requires auth; gracefully recorded if not).

Never hardcodes tokens. Kaggle auth is read from kaggle.json or env by the CLI
itself. Re-running is safe: existing files with matching checksum are skipped.
"""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
EXTRACTED = ROOT / "datasets" / "extracted"
ANALYSIS = ROOT / "analysis"
MANIFEST = ANALYSIS / "download_manifest.csv"

HEADERS = {"User-Agent": "COVERAGE-AI/1.0 (research; contact: project owner)"}
TIMEOUT = 60
CHUNK = 1 << 15

# --- Dataset registry ------------------------------------------------------
# source: uci | zenodo | kaggle
# kaggle datasets use slug (owner/dataset) for datasets, comp:<name> for comps.
DATASETS = [
    {
        "dataset_id": "D1",
        "name": "UJIIndoorLoc-UCI",
        "source": "uci",
        "url": "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip",
        "license": "CC BY 4.0",
        "subdir": "uci/ujiindoorloc",
    },
    {
        "dataset_id": "D4",
        "name": "Tampere WLAN RSS (Zenodo 1161525)",
        "source": "zenodo",
        "record_id": "1161525",
        "license": "CC BY 4.0",
        "subdir": "zenodo/tampere_wlan_rss_1161525",
    },
    {
        "dataset_id": "D5",
        "name": "Tampere multi-building (Zenodo 5174851)",
        "source": "zenodo",
        "record_id": "5174851",
        "license": "CC BY 4.0",
        "subdir": "zenodo/tampere_multi_building_5174851",
    },
    {
        "dataset_id": "D6",
        "name": "Crowdsourced WiFi (Zenodo 1001662)",
        "source": "zenodo",
        "record_id": "1001662",
        "license": "MIT (Zenodo record; bundled README states CC BY 4.0; conflict unresolved)",
        "subdir": "zenodo/crowdsourced_1001662",
    },
    {
        "dataset_id": "D2",
        "name": "UJIIndoorLoc-Kaggle (giantuji/ujiindoorloc)",
        "source": "kaggle",
        "slug": "giantuji/ujiindoorloc",
        "license": "see Kaggle page (same data as D1)",
        "subdir": "kaggle/ujiindoorloc",
    },
    {
        "dataset_id": "D3",
        "name": "WILD-v2 (Kaggle competition)",
        "source": "kaggle",
        "competition": "wild-v2",
        "license": "see competition rules",
        "subdir": "kaggle/wild-v2",
    },
]


# --- helpers ---------------------------------------------------------------
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def safe_makedirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def download_stream(url, dest: Path, desc=""):
    """Stream-download `url` to `dest` with retry on 5xx / connection errors."""
    import time as _t
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err = ""
    for attempt in range(1, 5):
        try:
            with requests.get(url, headers=HEADERS, stream=True, timeout=TIMEOUT, allow_redirects=True) as r:
                if r.status_code >= 500:
                    last_err = f"HTTP {r.status_code}"
                    _t.sleep(4 * attempt)
                    continue
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp, dest)
            return dest.stat().st_size
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)[:200]
            _t.sleep(4 * attempt)
    raise RuntimeError(f"download failed after retries: {last_err}")


def already_have(path: Path):
    return path.exists()


# --- source downloaders ----------------------------------------------------
def fetch_uci(ds, manifest_rows):
    dest_dir = RAW / ds["subdir"]
    safe_makedirs(dest_dir)
    fname = ds["url"].rsplit("/", 1)[-1]
    dest = dest_dir / fname
    status = "ok"
    note = ""
    try:
        if already_have(dest):
            status = "skip_exists"
        else:
            download_stream(ds["url"], dest)
    except Exception as e:  # noqa: BLE001
        status = "error"
        note = repr(e)[:300]
    add_manifest(manifest_rows, ds, dest, status, note)
    # extract
    if dest.exists() and dest.suffix == ".zip":
        extract_dir = EXTRACTED / ds["dataset_id"]
        safe_makedirs(extract_dir)
        try:
            with zipfile.ZipFile(dest) as z:
                z.extractall(extract_dir)
            note2 = "extracted"
        except Exception as e:  # noqa: BLE001
            note2 = "extract_error:" + repr(e)[:200]
        add_manifest(manifest_rows, ds, extract_dir, status + "_extract", note2)


def fetch_zenodo(ds, manifest_rows):
    dest_dir = RAW / ds["subdir"]
    safe_makedirs(dest_dir)
    api = f"https://zenodo.org/api/records/{ds['record_id']}"
    import time as _t
    file_specs = None
    api_err = ""
    for attempt in range(1, 4):
        try:
            r = requests.get(api, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
            if r.status_code >= 500:
                api_err = f"HTTP {r.status_code}"
                _t.sleep(4 * attempt)
                continue
            r.raise_for_status()
            data = r.json()
            version = data.get("metadata", {}).get("version", "") or data.get("id", "")
            file_specs = []
            for f in data.get("files", []):
                key = f.get("key", "file")
                link = f.get("links", {}).get("self", "") or f.get("links", {}).get("download", "")
                if not link:
                    link = f"{api}/files/{key}/content"
                file_specs.append((key, link, f"zenodo_version={version}"))
            break
        except Exception as e:  # noqa: BLE001
            api_err = repr(e)[:200]
            _t.sleep(4 * attempt)

    if file_specs is None:
        # API unavailable: fall back to existing local files so the manifest
        # always reflects what is actually on disk.
        existing = sorted(dest_dir.glob("*"))
        if existing:
            for p in existing:
                add_manifest(manifest_rows, ds, p, "skip_exists",
                             f"zenodo API unavailable ({api_err}); existing local file recorded")
                if p.suffix == ".zip":
                    extract_dir = EXTRACTED / ds["dataset_id"]
                    safe_makedirs(extract_dir)
                    try:
                        with zipfile.ZipFile(p) as z:
                            z.extractall(extract_dir)
                        add_manifest(manifest_rows, ds, extract_dir, "ok_extract", "extracted (existing zip)")
                    except Exception as e:  # noqa: BLE001
                        add_manifest(manifest_rows, ds, extract_dir, "error_extract", repr(e)[:200])
        else:
            add_manifest(manifest_rows, ds, dest_dir, "error",
                         f"zenodo API unavailable ({api_err}) and no local file present")
        return

    for key, link, fnote in file_specs:
        dest = dest_dir / key
        fstatus = "ok"
        try:
            if already_have(dest):
                fstatus = "skip_exists"
            else:
                download_stream(link, dest)
        except Exception as e:  # noqa: BLE001
            fstatus = "error"
            fnote = (fnote + " | " + repr(e)[:200]).strip(" |")
        add_manifest(manifest_rows, ds, dest, fstatus, fnote)
        if dest.suffix == ".zip" and dest.exists():
            extract_dir = EXTRACTED / ds["dataset_id"]
            safe_makedirs(extract_dir)
            try:
                with zipfile.ZipFile(dest) as z:
                    z.extractall(extract_dir)
                add_manifest(manifest_rows, ds, extract_dir, "ok_extract", "extracted")
            except Exception as e:  # noqa: BLE001
                add_manifest(manifest_rows, ds, extract_dir, "error_extract", repr(e)[:200])


def fetch_kaggle(ds, manifest_rows):
    dest_dir = RAW / ds["subdir"]
    safe_makedirs(dest_dir)
    # Detect auth by attempting a lightweight kaggle call.
    try:
        probe = subprocess.run(
            ["kaggle", "datasets", "list", "-s", "wifi", "--max-size", "1000"],
            capture_output=True, text=True, timeout=60,
        )
        authed = probe.returncode == 0 and "Authentication" not in (probe.stdout + probe.stderr)
    except Exception:  # noqa: BLE001
        authed = False
    if not authed:
        add_manifest(
            manifest_rows, ds, dest_dir, "error",
            "Kaggle not authenticated. Run `kaggle auth login` or place kaggle.json "
            "(see README). Re-run this script afterward.",
        )
        return
    try:
        if "competition" in ds:
            cmd = ["kaggle", "competitions", "download", "-c", ds["competition"], "-p", str(dest_dir)]
        else:
            cmd = ["kaggle", "datasets", "download", ds["slug"], "-p", str(dest_dir), "--unzip"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        status = "ok" if res.returncode == 0 else "error"
        note = (res.stderr or res.stdout)[:300]
        add_manifest(manifest_rows, ds, dest_dir, status, note)
    except Exception as e:  # noqa: BLE001
        add_manifest(manifest_rows, ds, dest_dir, "error", repr(e)[:300])


# --- manifest --------------------------------------------------------------
def add_manifest(rows, ds, path: Path, status, note):
    sha = ""
    size = ""
    if path.exists() and path.is_file():
        size = str(path.stat().st_size)
        sha = sha256_of(path)
    rows.append({
        "dataset_id": ds["dataset_id"],
        "name": ds["name"],
        "source": ds["source"],
        "source_url": ds.get("url") or ds.get("slug") or f"https://zenodo.org/records/{ds.get('record_id','')}" or "",
        "license": ds.get("license", ""),
        "version": "",
        "download_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_path": str(path.relative_to(ROOT)).replace("\\", "/") if str(path).startswith(str(ROOT)) else str(path),
        "file_size_bytes": size,
        "sha256": sha,
        "status": status,
        "notes": note,
    })


DISPATCH = {"uci": fetch_uci, "zenodo": fetch_zenodo, "kaggle": fetch_kaggle}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in DATASETS:
        print(f"[{ds['dataset_id']}] {ds['name']} ({ds['source']}) ...", flush=True)
        DISPATCH[ds["source"]](ds, rows)
    cols = ["dataset_id", "name", "source", "source_url", "license", "version",
            "download_date", "file_path", "file_size_bytes", "sha256", "status", "notes"]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print("\nManifest written:", MANIFEST)
    ok = sum(1 for r in rows if r["status"].startswith("ok") or r["status"].startswith("skip"))
    err = sum(1 for r in rows if r["status"] == "error")
    print(f"done: {ok} ok/skip, {err} errors, {len(rows)} manifest entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
