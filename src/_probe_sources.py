"""Probe real download endpoints (UCI static + Zenodo REST API).

This does NOT download the data. It only queries metadata so the dataset
inventory is built from verified facts instead of guesses. Prints a JSON
summary to stdout.
"""
import json
import sys
import requests

TIMEOUT = 30
HEADERS = {"User-Agent": "COVERAGE-AI/1.0 (research; contact: project owner)"}

UCI_STATIC = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"
ZENODO_RECORDS = {
    "tampere_wifi_rss": "1161525",
    "tampere_multi_building": "5174851",
    "crowdsourced_wifi": "1001662",
}
ZENODO_API = "https://zenodo.org/api/records/{recid}"


def probe_uci():
    out = {"url": UCI_STATIC}
    try:
        r = requests.head(UCI_STATIC, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        out["status"] = r.status_code
        out["content_type"] = r.headers.get("Content-Type", "")
        out["content_length"] = r.headers.get("Content-Length", "")
        out["final_url"] = r.url
        out["ok"] = r.ok
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["error"] = repr(e)
    return out


def probe_zenodo(recid):
    out = {"record_id": recid, "api_url": ZENODO_API.format(recid=recid)}
    try:
        r = requests.get(out["api_url"], headers=HEADERS, timeout=TIMEOUT)
        out["status"] = r.status_code
        out["ok"] = r.ok
        if r.ok:
            data = r.json()
            md = data.get("metadata", {})
            out["title"] = md.get("title", "")
            out["description_head"] = (md.get("description", "")[:300])
            out["doi"] = data.get("doi", "")
            out["license"] = (md.get("license", {}) or {}).get("id", "") if isinstance(md.get("license"), dict) else md.get("license", "")
            out["publication_date"] = md.get("publication_date", "")
            out["files"] = []
            total = 0
            for f in data.get("files", []):
                key = f.get("key", "")
                size = f.get("size", 0)
                total += size
                out["files"].append({
                    "key": key,
                    "size_bytes": size,
                    "checksum": f.get("checksum", ""),
                    "link": f.get("links", {}).get("self", "") or f.get("links", {}).get("download", ""),
                })
            out["total_size_bytes"] = total
            out["num_files"] = len(out["files"])
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["error"] = repr(e)
    return out


def main():
    result = {"uci": probe_uci(), "zenodo": {}}
    for name, recid in ZENODO_RECORDS.items():
        result["zenodo"][name] = probe_zenodo(recid)
    print(json.dumps(result, indent=2))
    return 0 if (result["uci"]["ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
