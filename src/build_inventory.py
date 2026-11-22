"""Build the dataset inventory and comparison CSVs from verified metadata.

Metadata below is taken from the UCI dataset page and from Zenodo's REST API
(probed by `_probe_sources.py`). Numbers are the published record counts /
column counts; they are facts about the datasets, not measured here.
"""
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
ANALYSIS.mkdir(parents=True, exist_ok=True)

# id, name, source, source_url, license, records_train, records_test,
# num_features, coord_info, rssi_info, bssid_ap_id, building_floor,
# environment, missing_value_repr, relevance_class, limitations,
# recommended_ml, download_priority
INVENTORY_COLS = [
    "dataset_id", "name", "source", "source_url", "license",
    "records_train", "records_test", "num_features", "coord_info",
    "rssi_info", "ap_identifier", "building_floor", "environment",
    "missing_value_repr", "relevance_class", "limitations",
    "recommended_ml", "download_priority", "accessible_now",
]

INVENTORY_ROWS = [
    {
        "dataset_id": "D1",
        "name": "UJIIndoorLoc (UCI)",
        "source": "UCI",
        "source_url": "https://archive.ics.uci.edu/dataset/310/ujiindoorloc",
        "license": "CC BY 4.0",
        "records_train": "19937",
        "records_test": "1111",
        "num_features": "529 (520 WAP + 9 location/context)",
        "coord_info": "Projected planar LONGITUDE/LATITUDE columns; observed ranges about -7691..-7301 and 4864746..4865017; source CRS/origin needs confirmation",
        "rssi_info": "dBm per WAP; +100 = not detected",
        "ap_identifier": "WAP001..WAP520 (anonymized AP index)",
        "building_floor": "BUILDINGID, FLOOR, SPACEID",
        "environment": "indoor, university campus (UJI, Spain)",
        "missing_value_repr": "+100 (bogus; means not detected)",
        "relevance_class": "A, B",
        "limitations": "coordinates are not WGS84 degrees; confirm source CRS and transform/translate to device-local east/north metres; single campus; domain shift vs ESP32",
        "recommended_ml": "RSSI prediction (long format), indoor localization",
        "download_priority": "1 (primary accessible)",
        "accessible_now": "yes",
    },
    {
        "dataset_id": "D2",
        "name": "UJIIndoorLoc (Kaggle mirror)",
        "source": "Kaggle",
        "source_url": "https://www.kaggle.com/datasets/giantuji/ujiindoorloc",
        "license": "see Kaggle page (same as D1)",
        "records_train": "19937",
        "records_test": "1111",
        "num_features": "529",
        "coord_info": "as D1",
        "rssi_info": "dBm; +100 = not detected",
        "ap_identifier": "WAP001..WAP520",
        "building_floor": "BUILDINGID, FLOOR",
        "environment": "indoor, university campus",
        "missing_value_repr": "+100",
        "relevance_class": "A, B",
        "limitations": "requires Kaggle auth; duplicate of D1",
        "recommended_ml": "same as D1",
        "download_priority": "3 (auth required)",
        "accessible_now": "no (Kaggle auth missing)",
    },
    {
        "dataset_id": "D3",
        "name": "WILD-v2 (Kaggle competition)",
        "source": "Kaggle",
        "source_url": "https://www.kaggle.com/competitions/wild-v2/data",
        "license": "see competition rules",
        "records_train": "see Kaggle page",
        "records_test": "see Kaggle page",
        "num_features": "Wi-Fi + sensor fingerprints",
        "coord_info": "floor / building context",
        "rssi_info": "RSSI fingerprints",
        "ap_identifier": "AP identifiers per competition",
        "building_floor": "yes (multi-floor)",
        "environment": "indoor",
        "missing_value_repr": "per file (inspect after download)",
        "relevance_class": "B",
        "limitations": "requires Kaggle auth; localization-focused",
        "recommended_ml": "indoor localization",
        "download_priority": "3 (auth required)",
        "accessible_now": "no (Kaggle auth missing)",
    },
    {
        "dataset_id": "D4",
        "name": "Tampere WLAN RSS (Zenodo 1161525)",
        "source": "Zenodo",
        "source_url": "https://zenodo.org/records/1161525",
        "license": "CC BY 4.0",
        "records_train": "see files (Training/Test split)",
        "records_test": "see files",
        "num_features": "APs as columns (fingerprint matrix)",
        "coord_info": "X/Y coordinates (units confirmed at inspection)",
        "rssi_info": "RSS per AP; convention inspected at load",
        "ap_identifier": "AP columns (index)",
        "building_floor": "single building (Tampere)",
        "environment": "indoor, university",
        "missing_value_repr": "inspect at load (typically +100 or NaN)",
        "relevance_class": "A, B",
        "limitations": "smaller; single environment; domain shift vs ESP32",
        "recommended_ml": "RSSI prediction, localization",
        "download_priority": "2",
        "accessible_now": "yes",
    },
    {
        "dataset_id": "D5",
        "name": "Tampere multi-building (Zenodo 5174851)",
        "source": "Zenodo",
        "source_url": "https://zenodo.org/records/5174851",
        "license": "CC BY 4.0",
        "records_train": "TIE1=10633 + SAH1=9291",
        "records_test": "TIE1=50 + SAH1=156",
        "num_features": "TIE1=613 APs; SAH1=775 APs",
        "coord_info": "X, Y, Z metres (local) + floor",
        "rssi_info": "dBm per AP; +100 = not detected",
        "ap_identifier": "AP columns (index)",
        "building_floor": "TIE1 (Tietotalo), SAH1 (Sahkotalo), floor",
        "environment": "indoor, Tampere University campus (2017)",
        "missing_value_repr": "+100 (documented bogus = not detected)",
        "relevance_class": "A (primary)",
        "limitations": "smartphone-collected; domain shift vs ESP32; still best coordinate fit",
        "recommended_ml": "RSSI prediction (primary), localization",
        "download_priority": "1 (PRIMARY training set)",
        "accessible_now": "yes",
    },
    {
        "dataset_id": "D6",
        "name": "Crowdsourced WiFi (Zenodo 1001662)",
        "source": "Zenodo",
        "source_url": "https://zenodo.org/records/1001662",
        "license": "MIT (Zenodo record; bundled README states CC BY 4.0; conflict unresolved)",
        "records_train": "see zip contents",
        "records_test": "see zip contents",
        "num_features": "crowdsourced Wi-Fi fingerprints",
        "coord_info": "inspect after extraction",
        "rssi_info": "RSSI fingerprints",
        "ap_identifier": "AP identifiers in files",
        "building_floor": "variable (crowdsourced)",
        "environment": "indoor, heterogeneous crowdsourced",
        "missing_value_repr": "inspect after extraction",
        "relevance_class": "B, D",
        "limitations": "heterogeneous collection -> high domain shift; localization/propagation reference",
        "recommended_ml": "localization, propagation reference",
        "download_priority": "3",
        "accessible_now": "yes",
    },
]

COMPARISON_COLS = [
    "dataset_id", "name", "rssi_prediction_suitability",
    "localization_suitability", "coverage_classification_suitability",
    "propagation_research_suitability", "coord_matches_esp32",
    "not_detected_convention", "class", "notes",
]
COMPARISON_ROWS = [
    {"dataset_id": "D5", "name": "Tampere multi-building", "rssi_prediction_suitability": "high", "localization_suitability": "high", "coverage_classification_suitability": "high (derived)", "propagation_research_suitability": "medium", "coord_matches_esp32": "yes (X/Y metres)", "not_detected_convention": "+100 dBm", "class": "A", "notes": "PRIMARY: metre coords + documented +100 + 2 buildings for spatial splits"},
    {"dataset_id": "D1", "name": "UJIIndoorLoc-UCI", "rssi_prediction_suitability": "high", "localization_suitability": "high", "coverage_classification_suitability": "high (derived)", "propagation_research_suitability": "medium", "coord_matches_esp32": "partial (projected planar coords; source CRS/origin needs confirmation)", "not_detected_convention": "+100 dBm", "class": "A,B", "notes": "large, well-documented; coordinates are not WGS84 degrees"},
    {"dataset_id": "D4", "name": "Tampere WLAN RSS", "rssi_prediction_suitability": "medium-high", "localization_suitability": "high", "coverage_classification_suitability": "medium", "propagation_research_suitability": "medium", "coord_matches_esp32": "likely yes (verify units)", "not_detected_convention": "inspect", "class": "A,B", "notes": "smaller single-building fingerprint set"},
    {"dataset_id": "D2", "name": "UJIIndoorLoc-Kaggle", "rssi_prediction_suitability": "high", "localization_suitability": "high", "coverage_classification_suitability": "high (derived)", "propagation_research_suitability": "medium", "coord_matches_esp32": "partial", "not_detected_convention": "+100 dBm", "class": "A,B", "notes": "mirror of D1; needs Kaggle auth"},
    {"dataset_id": "D3", "name": "WILD-v2", "rssi_prediction_suitability": "medium", "localization_suitability": "high", "coverage_classification_suitability": "medium", "propagation_research_suitability": "low", "coord_matches_esp32": "unknown", "not_detected_convention": "inspect", "class": "B", "notes": "competition; needs Kaggle auth"},
    {"dataset_id": "D6", "name": "Crowdsourced WiFi", "rssi_prediction_suitability": "low", "localization_suitability": "medium", "coverage_classification_suitability": "low", "propagation_research_suitability": "medium", "coord_matches_esp32": "yes (local X/Y metres)", "not_detected_convention": "+100 dBm", "class": "B,D", "notes": "heterogeneous; Zenodo says MIT but bundled README says CC BY 4.0; resolve before redistribution"},
]


def write_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    write_csv(ANALYSIS / "dataset_inventory.csv", INVENTORY_COLS, INVENTORY_ROWS)
    write_csv(ANALYSIS / "dataset_comparison.csv", COMPARISON_COLS, COMPARISON_ROWS)


if __name__ == "__main__":
    main()
