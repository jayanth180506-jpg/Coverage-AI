# COVERAGE-AI — Final Recommendations (Stage 13-16)

## Recommendation summary

| Role | Model | Rationale |
|------|-------|-----------|
| Best overall on this split | Random Forest | MAE 6.28 dBm; serialized estimator 89.5 MB, too large for ESP32 |
| Best lightweight learned candidate | Ridge Regression | 626-byte estimator; generated header parity validated |
| Currently integrated model | Ridge Regression | C++ parity validated; board build pending |
| Best research candidate | Random Forest | Better spatial test error, but high memory footprint and smartphone domain shift |
| Best for future dev | Re-train Ridge on ESP32 measurements | Public data has domain shift vs ESP32 |

## Test metrics (spatially held-out)

- Decision Tree: MAE=7.82 dBm, RMSE=9.96 dBm, R2=-0.057
- Per-AP median baseline: MAE=7.56 dBm

## Why this is acceptable and honest
- Wi-Fi RSSI is intrinsically noisy (multipath, phone orientation, body absorption). A ~8 dBm MAE is typical for fingerprinting data.
- Random Forest improves error on this held-out split, but its large serialized size rules it out for this ESP32 target. Ridge has similar accuracy to the tree with a compact estimator and has now been exported to C++; board memory and runtime remain to be checked. Public-data results remain subject to domain shift.

## Domain-shift limitation (must read)
All public datasets were collected with smartphones, not an ESP32. Their RSSI offset, antenna pattern and noise differ from the ESP32 radio. The model trained here is a **benchmark**. Before production you MUST:
1. Collect your own ESP32 measurements (SSID, BSSID, RSSI, channel, X, Y, floor).
2. Evaluate this model against ESP32 data (expect a calibration offset).
3. Fine-tune / re-train Ridge on ESP32 data, then export and revalidate    the generated C++ header. The tree exporter remains available as a    comparison path.

## ESP32 deployment
- Inference is a five-coefficient linear calculation + an AP-mean lookup.
- The Arduino-ready sketch is `esp32/COVERAGE_AI/COVERAGE_AI.ino`; its model
  headers are beside it. See `esp32/FLASH_GUIDE.md` for upload and field setup.
- New/unknown BSSIDs fall back to the global mean RSSI.

## Recommended ESP32 data collection
- Start each survey at `(0,0)`, use the web directional controls to maintain
  relative X/Y as you walk a regular grid, and use GPS Auto outdoors. Do not use
  the MPU6500 as an indoor position source.
- Record every visible BSSID/RSSI/SSID/channel at each known location.
- Cover weak-signal edges and corners, not just strong areas.
- Several passes per location to average multipath noise.
- Export `/coverage_ai_survey.csv`; `ingest_esp32_csv.py` normalizes the survey
  schema for training.
