# COVERAGE-AI — Dataset Discovery & Suitability Report

**Stage:** 1–2 (Discovery, Inventory, Suitability Comparison)
**Date generated:** auto (see manifest `download_date`)
**Author role:** Senior ML / Wireless / Embedded engineering

---

## 1. Objective

COVERAGE-AI must predict Wi-Fi **RSSI (dBm)** at unmeasured physical locations
from coordinates (X east-west metres, Y north-south metres), access-point
identity and environment metadata, then run inference on a standalone ESP32
(TTGO T-Call V1.4). This report catalogs public datasets relevant to that goal
and classifies their suitability.

---

## 2. Search scope

Kaggle search terms used (conceptually, since the API is not authenticated in
this environment — see §5 limitation):

`Wi-Fi RSSI`, `Wi-Fi signal strength`, `Wi-Fi fingerprinting`,
`Indoor Wi-Fi localization`, `Indoor positioning`, `Wireless signal strength
prediction`, `Wi-Fi coverage mapping`, `WLAN fingerprinting`,
`Access point localization`, `Wireless network measurement`,
`Wi-Fi propagation`, `Indoor radio map`, `RSSI-based localization`,
`Wireless sensor localization`.

Specific sources investigated: UJIIndoorLoc (Kaggle + UCI), WILD-v2 (Kaggle),
Tampere University Wi-Fi RSS datasets (Zenodo ×2), Crowdsourced Wi-Fi (Zenodo).

---

## 3. Candidate datasets (verified metadata)

### 3.1 UJIIndoorLoc — UCI (primary accessible copy)
- **Source:** https://archive.ics.uci.edu/dataset/310/ujiindoorloc
- **Static zip:** https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip
- **License:** CC BY 4.0 (UJI / Torres-Sospedra et al.)
- **Records:** 19,937 training + 1,111 validation fingerprints.
- **Features:** 520 WAP columns (RSSI per access point) + 9 location/context
  columns: LONGITUDE, LATITUDE, FLOOR, BUILDINGID, SPACEID, RELATIVEPOSITION,
  USERID, PHONEID, TIMESTAMP.
- **RSSI representation:** dBm; **not-detected = +100** (documented bogus value).
- **Coordinates:** columns named LONGITUDE/LATITUDE contain projected planar
  values (observed approximately -7691..-7301 and 4864746..4865017), not WGS84
  degree values. The source CRS and origin still need confirmation; translate or
  transform them into the COVERAGE-AI local east/north frame before reuse.
- **Environment:** indoor, university campus (Universitat Jaume I, Spain).
- **Relevance:** A. RSSI prediction (long-format reshape), B. localization.
- **Limitations:** projected coordinate CRS/origin needs confirmation; single
  campus → domain shift vs ESP32; +100 convention must be
  treated as missing, never as RSSI.

### 3.2 UJIIndoorLoc — Kaggle mirror
- **Source:** https://www.kaggle.com/datasets/giantuji/ujiindoorloc
- **Same data as 3.1;** license per Kaggle page.
- **Status:** Requires Kaggle authentication (not configured here). Script
  will download automatically once `kaggle auth login` or a `kaggle.json` is
  present (see README). UCI copy is used as the accessible equivalent.

### 3.3 WILD-v2 — Kaggle competition
- **Source:** https://www.kaggle.com/competitions/wild-v2/data
- **Content:** Wi-Fi + other-sensor fingerprints for localization; multi-floor.
- **Status:** Requires Kaggle authentication. Included in download script.
- **Relevance:** B. localization; secondary for RSSI once downloaded.

### 3.4 Tampere WLAN RSS — Zenodo 1161525 (single dataset)
- **Source:** https://zenodo.org/records/1161525 — DOI 10.5281/zenodo.1161525
- **License:** CC BY 4.0
- **Files (verified via Zenodo API):** `Training_rss.csv` (922 KB),
  `Training_coordinates.csv`, `Test_rss.csv` (1.9 MB), `Test_coordinates.csv`.
- **Structure:** RSS fingerprint matrices (rows = reference samples,
  columns = APs) + matching X/Y coordinates.
- **Relevance:** A. RSSI prediction, B. localization.
- **Limitations:** smaller; coordinate units to be confirmed at inspection.

### 3.5 Tampere multi-building campus — Zenodo 5174851 (PRIMARY)
- **Source:** https://zenodo.org/records/5174851 — DOI 10.5281/zenodo.5174851
- **License:** CC BY 4.0 — Lohan, Torres-Sospedra, Gonzalez (2021).
- **Files (verified):** single zip `... 5174851.zip` (1.48 MB, md5
  `9ed418ecea09890692ea8a464895395a`) containing:
  - **TIE1** (Tietotalo): 10,633 training + 50 test samples, 613 APs.
  - **SAH1** (Sahkotalo): 9,291 training + 156 test samples, 775 APs.
- **Coordinates:** **X, Y, Z in metres in a local system + floor** — this is a
  direct match for COVERAGE-AI's relative coordinate definition.
- **RSSI representation:** dBm; **not-detected = +100 dBm** (documented, same
  convention as UJIIndoorLoc).
- **Relevance:** **A. RSSI prediction (primary)** — best coordinate fit.
- **Why primary:** metres-based local coordinates match the ESP32 device, the
  +100 convention is documented, and two buildings enable spatial/group-based
  train/validation/test splits.

### 3.6 Crowdsourced Wi-Fi — Zenodo 1001662
- **Source:** https://zenodo.org/records/1001662
- **License:** Zenodo record metadata lists MIT, while the bundled
  `FINGERPRINTING_DB/README.txt` says CC BY 4.0. This conflict is unresolved;
  confirm the authoritative license before redistribution.
- **Files (verified):** `DISTRIBUTED_OPENSOURCE_version2.zip` (685 KB).
- **Content:** crowdsourced Wi-Fi fingerprint database + benchmark software.
- **Relevance:** B. localization; lower priority for RSSI prediction due to
  heterogeneous collection (inherent domain shift).

---

## 4. Suitability classification

| ID | Dataset | Class | Role in COVERAGE-AI |
|----|---------|-------|--------------------|
| D5 | Tampere multi-building (5174851) | A | **Primary RSSI-prediction training set** |
| D1 | UJIIndoorLoc-UCI (310) | A, B | RSSI prediction + localization; secondary |
| D4 | Tampere WLAN RSS (1161525) | A, B | Secondary RSSI / localization |
| D2 | UJIIndoorLoc-Kaggle (giantuji) | A, B | Mirror of D1 (needs auth) |
| D3 | WILD-v2 (Kaggle) | B | Localization (needs auth) |
| D6 | Crowdsourced (1001662) | B, D | Localization / propagation reference |

**A** = directly useful for RSSI prediction; **B** = localization; **C** =
coverage classification (derived, none primary); **D** = propagation research;
**E** = not suitable. No dataset was classified E.

---

## 5. Limitations & assumptions

1. **Kaggle not authenticated.** Datasets D2 (giantuji/ujiindoorloc) and D3
   (WILD-v2) cannot be downloaded in this environment because there is no
   `kaggle.json` and no `KAGGLE_USERNAME`/`KAGGLE_KEY`. The download script uses
   the Kaggle API and will fetch them automatically once credentials are
   configured (instructions in README). UCI provides an accessible copy of
   UJIIndoorLoc, so the project is not blocked.
2. **Domain shift.** All public datasets were collected with smartphones / Wi-Fi
   cards differing from the ESP32. RSSI distributions will not match the device.
   Models trained here are benchmarks; production readiness requires ESP32
   measurement collection and fine-tuning (see final_recommendations.md).
3. **+100 dBm convention.** UJIIndoorLoc and Tampere use +100 for "not
   detected". This is treated as a **missing** value, never as a real RSSI.
   Concrete missing-value strategy is documented in preprocessing_report.md.

---

## 6. Decision

Proceed to download the four accessible datasets (D1, D4, D5, D6) and attempt
the two Kaggle datasets (D2, D3). Use **D5 (Tampere multi-building)** as the
primary RSSI-prediction training set because its metre-based local coordinates
and documented +100 convention align with the ESP32 device definition.
