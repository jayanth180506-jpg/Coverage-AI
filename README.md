# COVERAGE-AI — ML Pipeline & ESP32 Deployment

A complete, reproducible machine-learning pipeline for **COVERAGE-AI**, an
ESP32-based (TTGO T-Call V1.4) Wi-Fi coverage measurement, visualization,
prediction and optimization system. The laptop is used **only** for dataset
preparation, training, evaluation, model export and firmware development; the
device runs independently.

The primary ML objective is **RSSI prediction**: predict Wi-Fi RSSI (dBm) at a
physical location for a given access point, then run that inference on the
ESP32 as lightweight C++.

---

## 1. What this pipeline does (stage by stage)

| Stage | Script / output | Description |
|------:|-----------------|-------------|
| 1-2 | `src/build_inventory.py` -> `analysis/dataset_inventory.csv`, `analysis/dataset_comparison.csv`, `reports/dataset_report.md` | Discover & classify public datasets |
| 3   | `src/download_datasets.py` -> `datasets/raw/`, `datasets/extracted/`, `analysis/download_manifest.csv` | Download (Kaggle API + UCI + Zenodo), checksum, extract |
| 4   | `src/inspect_datasets.py` -> `analysis/dataset_quality_report.csv`, `analysis/figures/`, `reports/inspection_report.md` | Inspect every dataset (sentinel, coords, APs, missing) |
| 5-6 | `src/preprocess_all.py` -> `datasets/processed/*_long.csv`, `reports/preprocessing_report.md` | Common long format for source train/validation/test splits; +100 is dropped as missing |
| 7-8 | (inside `train_models.py`) -> `analysis/feature_analysis.csv` | EDA + feature engineering (target-encoded AP) |
| 9-11| `src/train_models.py` -> `models/trained/model_bundle.pkl`, `models/comparison/model_metrics.csv` | Spatial holdout; tree by default, full regressor benchmark with `--full` |
| 9-11| `src/train_coverage_classifier.py`, `src/train_localization.py`, `src/identify_weak_coverage.py` | Separate coverage classification, indoor localization, weak-region objectives |
| 11  | `src/evaluate_models.py` -> `reports/model_comparison.md`, `reports/final_recommendations.md`, `models/comparison/*.png` | Evaluate, plots, selection |
| 13  | `src/export_esp32_model.py` -> `models/exported/wifi_rssi_model.h`, `test_inputs.csv`, `test_model.cpp` | Export tree to C++ |
| 14  | `src/validate_export.py` -> `reports/export_validation.md` | Compile with g++, compare C++ vs Python |
| 14  | `src/export_ridge_esp32.py`, `src/validate_ridge_export.py` | Export and validate compact Ridge C++ inference |
| 15  | `esp32/COVERAGE_AI/COVERAGE_AI.ino`, `wifi_rssi_ridge.h`, `wifi_rssi_calibration.h` | ESP32 Arduino sketch (survey + maps + inference + SD history) |
| 16  | `README.md`, `reports/*.md`, `notebooks/*.ipynb` | Documentation & future data-collection plan |

---

## 2. Environment

Tested on **Windows + Python 3.11.9**. Required packages (already satisfied in
this environment; pinned in `requirements.txt`):

```
pandas numpy scipy scikit-learn matplotlib seaborn joblib requests kaggle
```

Install fresh:
```powershell
python -m pip install -r requirements.txt
```

A C++ compiler is needed only for Stage 14 (Python-vs-C++ validation). This
environment uses **MinGW g++ 6.3.0** (`C:\MinGW\bin\g++.exe`). For ESP32 firmware
you additionally need the **Arduino IDE + ESP32 board support**.

---

## 3. Quick start (run the whole pipeline)

From the `COVERAGE_AI_ML` directory:

```powershell
python src\build_inventory.py        # dataset inventory + comparison CSVs
python src\download_datasets.py       # download accessible datasets + manifest
python src\inspect_datasets.py        # inspect + quality report + figures
python src\preprocess_all.py          # all datasets and separate official splits
python src\train_models.py --dataset D5 --full  # spatially compare required regressors
python src\train_coverage_classifier.py --dataset D5
python src\train_localization.py      # separate UJI Wi-Fi fingerprint -> X/Y model
python src\evaluate_models.py         # evaluation plots + reports
python src\evaluate_official_splits.py
python src\evaluate_d4_official.py
python src\evaluate_localization_validation.py
python src\identify_weak_coverage.py --threshold -70
python src\export_esp32_model.py      # export tree -> C++ header
python src\validate_export.py         # compile + validate C++ == Python
python src\export_ridge_esp32.py      # after full training
python src\validate_ridge_export.py
```

Or run the notebooks top-to-bottom in `notebooks/`.

---

## 4. Kaggle authentication (for D2 UJIIndoorLoc mirror & D3 WILD-v2)

Two datasets live only on Kaggle. The download script uses the official Kaggle
API, so they are fetched automatically once you authenticate:

```powershell
# Option A (recommended, no token file to manage):
kaggle auth login
# Option B: create a token at https://www.kaggle.com/settings/api (API section,
# "Create New Token"), save it as  %USERPROFILE%\.kaggle\kaggle.json
```
Then re-run `python src\download_datasets.py`. Until then the script records a
clear "Kaggle not authenticated" error in the manifest (not silently skipped).
UJIIndoorLoc is also downloaded directly from UCI (no auth), so the project is
not blocked.

---

## 5. Datasets used (verified)

| ID | Dataset | Source | Status | Role |
|----|---------|--------|--------|------|
| D1 | UJIIndoorLoc | UCI | downloaded (CC BY 4.0) | secondary (projected planar coordinates; source CRS/origin needs confirmation) |
| D5 | Tampere multi-building | Zenodo 5174851 | downloaded (CC BY 4.0) | **PRIMARY** (metre coords + floor, 2 buildings) |
| D4 | Tampere WLAN RSS | Zenodo 1161525 | downloaded (CC BY 4.0) | secondary |
| D6 | Crowdsourced Wi-Fi | Zenodo 1001662 | downloaded (MIT) | localization/propagation reference |
| D2 | UJIIndoorLoc mirror | Kaggle | needs auth | mirror of D1 |
| D3 | WILD-v2 | Kaggle | needs auth | localization |

All fingerprint datasets use **+100 dBm = "access point not detected"**; the
preprocessor treats that as a missing value (dropped), never as RSSI.

---

## 6. Modeling design

- **Features available at ESP32 inference:** `x (m), y (m), floor_id,
  building_num, ap_mean_rssi`. BSSID is **not** encoded as an ordinal integer
  (spec rule); it is **target-encoded** as the AP's mean RSSI computed on the
  training fold (one compact feature + a small lookup table). Unseen APs fall
  back to the global mean.
- **Spatial split:** each building's (x,y) plane is binned into a 5x5 grid;
  whole cells are held out with `GroupShuffleSplit`, so the test set is genuinely
  unseen *areas* (not nearby correlated points). Depth is chosen by spatial
  `GroupKFold` CV.
- The default run trains the deployment tree. Add `--full` to evaluate global
  mean/median and per-AP mean/median baselines plus Linear Regression, Ridge,
  Random Forest, and Gradient Boosting on the same held-out spatial cells. The
  benchmark records error, training time, inference time, feature count, and
  serialized model size. Only the small Decision Tree is currently exported to
  C++; revisit model selection after running the full benchmark.

### Results (D5, spatial holdout)

| Model | MAE (dBm) | RMSE (dBm) | R² |
|-------|-----------|-----------|-----|
| Per-AP median baseline | 7.69 | 10.60 | -0.20 |
| **Decision Tree (depth 4)** | **7.82** | **9.96** | -0.06 |
| Per-AP trees (mean, top-15) | 8.39 | — | — |

Per building (Decision Tree, test): TIE1 MAE 7.26, SAH1 MAE 8.36.

**Interpretation (honest):** RSSI is intrinsically noisy. The AP's average level
is the dominant, robust predictor; precise (x,y) adds little *generalizable*
signal on public fingerprinting data (R² ≈ 0 on unseen areas). This supports
starting with a compact embedded model and shows real improvement requires dense, device-specific
ESP32 measurements (domain-shift caveat below). The complete benchmark now shows
Random Forest improves error (MAE 6.28 dBm, RMSE 8.54 dBm) at an 89.5 MB serialized
size, while Ridge reaches MAE 7.73 dBm in a 626-byte serialized estimator. The
Ridge model has now been exported to a 51.7 KB header (including 1,319 AP mean
entries) and its C++ predictions match scikit-learn within 1.4e-14 dBm. The
Arduino sketch now uses Ridge; a board-target compile and measured RAM/flash
check remain necessary.

### C++ export validation

`validate_ridge_export.py` compiles the Ridge C++ harness and compares 2000
predictions: **maximum absolute difference = 1.391e-14 dBm** (PASS).

---

## 7. ESP32 deployment

Open `esp32/COVERAGE_AI/COVERAGE_AI.ino` in Arduino IDE. That folder contains
the complete website-enabled firmware and both model headers in the layout
Arduino requires. The device hosts a `CoverageAI` network, serves the dashboard
at `http://192.168.4.1`, and scans continuously. Surveys start at local `(0,0)`
without GPS; pause and stop control recording, while scans continue. The
dashboard provides directional position controls, an expanding square map,
BSSID-isolated measured points, distance-masked IDW, optional embedded ML,
zoom/pan, point inspection, and session-aware CSV history. For board setup,
wiring, dashboard use, site retraining, and flashing instructions, follow
[`esp32/FLASH_GUIDE.md`](esp32/FLASH_GUIDE.md).

The device appends complete history to `/coverage_ai_survey.csv`. Each AP row
has its scan measurement ID and survey session ID; absent APs are not written
as fabricated weak readings. The offline pipeline accepts its `x,y` fields and
writes rejected rows separately. The exporter and calibration script update
headers in the Arduino sketch folder as well as the model's `esp32/` directory.
Public-data predictions remain a benchmark until validated against ESP32/site
measurements.

---

## 8. Limitations you must read

- **Domain shift:** public datasets were collected with smartphones, not an
  ESP32. The trained model is a **benchmark**. Collect ESP32 measurements,
  evaluate this model against them, then fine-tune/retrain and re-export.
- **Kaggle:** D2/D3 need authentication (see §4).
- **Spatial generalization:** the model predicts RSSI best via the AP average;
  exact-RSSI-at-location on unseen areas is hard (documented R² ≈ 0).

---

## 9. Directory map

```
COVERAGE_AI_ML/
  datasets/raw/{kaggle,uci,zenodo}  datasets/extracted  datasets/processed  datasets/rejected
  analysis/  (inventory, comparison, quality_report, feature_analysis, figures/, download_manifest.csv)
  notebooks/ (01..05 .ipynb)
  models/trained  models/exported  models/comparison
  reports/  (dataset_report, preprocessing_report, model_comparison, final_recommendations, inspection_report, export_validation)
  src/      (download, inspect, preprocess, train, evaluate, export, validate, build_inventory, _probe_*)
  esp32/    (COVERAGE_AI/ Arduino sketch, model headers, FLASH_GUIDE.md)
  requirements.txt  README.md
```

Reproducibility: every script is idempotent; re-running skips files already on
disk and regenerates manifests/reports from current data.
