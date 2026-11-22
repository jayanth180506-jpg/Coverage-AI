# COVERAGE-AI Completion Audit

Audit date: 2026-11-22

## Work now present

- Dataset inventory, suitability comparison, source manifest, structural
  inspection, and quality figures cover UJIIndoorLoc, Tampere, Crowdsourced
  Wi-Fi, and the Kaggle candidates. Kaggle D2/D3 still need credentials.
- `src/preprocess_all.py` emits separate D1 training/validation, D4 training/
  test, D5 training/test, and D6 filename-labeled outputs. The +100 AP-missing
  sentinel is dropped. D6 source names are preserved because its bundled
  README contradicts itself about split semantics.
- Five exploratory/training/evaluation notebooks are present. Scripts remain
  the reproducible source of truth.
- D5 full regression benchmark runs on a spatial holdout. Random Forest leads
  accuracy (MAE 6.28 dBm, RMSE 8.54 dBm) but serializes to 89.5 MB. Ridge is
  close to the tree (MAE 7.73 dBm) with a 626-byte estimator.
- Ridge exports to `wifi_rssi_ridge.h`; 2,000 Python/C++ predictions passed
  parity validation with maximum difference 1.391e-14 dBm. The Arduino sketch
  now uses the Ridge header.
- Coverage classification compares majority, Decision Tree, and Random Forest
  on held-out spatial cells. Random Forest scored 0.455 accuracy and 0.332
  macro-F1; the classification report and confusion matrix are saved.
- Indoor localization is evaluated separately on UJI WAP fingerprints with
  spatial cells held out. Median Euclidean error is 17.69 source coordinate
  units. On the untouched UJI publisher validation split, median Euclidean
  error is 11.19 coordinate units; the source CRS/origin remains unconfirmed.
- The D5 official test split is scored separately: Ridge MAE 5.78 dBm and
  Decision Tree MAE 6.56 dBm (9,780 AP readings). D4's separate official test
  evaluation reports Ridge MAE 7.59 dBm (34,179 readings).
- Weak coverage regions are summarized from the held-out D5 spatial
  predictions. At the initial -70 dBm setting, all 25 summarized bins have
  mean predicted RSSI below threshold; this is a coarse result from public data.
- The survey firmware now emits session-aware rows with one measurement ID per
  scan and distinct BSSID readings. `ingest_esp32_csv.py` accepts the survey
  schema, validates/normalizes it and saves rejected rows separately.
  `calibrate_esp32.py` estimates an additive offset on held-out locations and
  synchronizes its header into the Arduino-ready sketch folder.
- UJIIndoorLoc's LONGITUDE/LATITUDE fields are documented as projected planar
  values, not WGS84 degrees. Their source CRS/origin still requires
  confirmation before use in the device frame.

## Remaining external work

1. No genuine ESP32 collection CSV is available in the workspace. Device
   calibration and site-specific retraining cannot be reported until the user
   collects measurements with known coordinates across multiple locations.
2. `arduino-cli` and the Arduino IDE CLI are not installed here. The generated
   C++ inference was compiled and validated with g++, but the complete sketch
   has not been compiled for the TTGO T-Call or measured for RAM/flash use.
3. Kaggle D2/D3 remain unavailable until credentials are configured.
4. D6's license metadata differs between the Zenodo manifest and the bundled
   README. Resolve the authoritative license before redistribution.
5. Router placement recommendations remain out of scope without a validated
   site-specific spatial model and a defined optimization objective.

## Survey firmware implementation (2026-11-22)

- Manual relative coordinates are the default; each new session begins at
  `(0,0)`. Scanning does not require a position or a GPS fix. Direction buttons
  move the local position; GPS Auto is optional, rebases its origin to preserve
  the current coordinates, and retains the last good location on GPS loss.
- Start, pause, resume, stop and new-session controls gate logging while Wi-Fi
  scans continue. Each detected AP row is isolated by BSSID and carries session,
  measurement, timestamp, position source, channel and RSSI data.
- The SD card stores append-only history in `/coverage_ai_survey.csv`. The RAM
  ring contains the latest 500 readings for the live map; the dashboard lists
  recent sessions and can download the full log or one session.
- The dashboard uses a square, expanding coordinate frame with pan, zoom,
  reset and point inspection. The ESP32 serves separate BSSID-filtered measured
  points, IDW grids (power 2, exact-sample handling, 12 m distance mask) and an
  optional ML grid. IDW and ML grids are bounded at 24 x 24 cells.
- ESP32 CSV import compatibility, dashboard JavaScript syntax, Python source
  compilation, and the 2,000-row Ridge C++ parity validation passed. The full
  Arduino firmware could not be compiled here because Arduino CLI/PlatformIO
  and a connected board are unavailable.

### Remaining firmware limits

- The live heatmap shows the current session's recent RAM cache, not a rebuild
  of old SD sessions. Old session data is available through CSV downloads.
- Firmware behavior still needs a board build/upload, peripheral checks, and
  real survey validation. Model quality remains unverified for the user's site.

## Reproduction commands

From `COVERAGE_AI_ML`:

```powershell
python src\preprocess_all.py
python src\train_models.py --dataset D5 --full
python src\train_coverage_classifier.py --dataset D5
python src\train_localization.py
python src\evaluate_models.py
python src\identify_weak_coverage.py --threshold -70
python src\export_ridge_esp32.py
python src\validate_ridge_export.py
```

When real ESP32 data exists:

```powershell
python src\ingest_esp32_csv.py path\to\device.csv
python src\calibrate_esp32.py datasets\processed\ESP32_long.csv
python src\train_models.py --dataset ESP32 --full
python src\export_ridge_esp32.py
python src\validate_ridge_export.py
```

Calibration currently evaluates an additive offset against the public
benchmark bundle. The ESP32 dataset must include repeated AP readings at
multiple spatially distinct measurement locations. After retraining, keep the
resulting site model as a separate artifact rather than overwriting the public
benchmark used for comparison.
