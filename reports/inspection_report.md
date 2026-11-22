# COVERAGE-AI — Dataset Inspection Report (Stage 4)
- **D1_train** (UJIIndoorLoc training): 19937 samples × 520 APs; sentinel=100.0 (frac=0.965); RSSI mean=-78.54861041380772, median=-82.0
- **D1_val** (UJIIndoorLoc validation): 1111 samples × 520 APs; sentinel=100.0 (frac=0.968); RSSI mean=-77.68378496503496, median=-80.0
- **D4_train** (Tampere WLAN RSS training): 446 samples × 489 APs; sentinel=100.0 (frac=0.913); coords cols=['x', 'y', 'z']; RSSI mean=-73.33634775733938, median=-76.161
- **D4_test** (Tampere WLAN RSS test): 982 samples × 489 APs; sentinel=100.0 (frac=0.929); coords cols=['x', 'y', 'z']; RSSI mean=-71.50429044149917, median=-73.75
- **D5_TIE1_train** (Tampere TIE1 training): 10633 samples × 613 APs; sentinel=100.0 (frac=0.931); coords cols=['x', 'y', 'z', 'floor']; RSSI mean=-74.57060164531211, median=-77.0
- **D5_TIE1_test** (Tampere TIE1 test): 50 samples × 613 APs; sentinel=100.0 (frac=0.881); coords cols=['x', 'y', 'z', 'floor']; RSSI mean=-74.28469945355191, median=-76.0
- **D5_SAH1_train** (Tampere SAH1 training): 9291 samples × 775 APs; sentinel=100.0 (frac=0.958); coords cols=['x', 'y', 'z', 'floor']; RSSI mean=-74.84016359756839, median=-78.0
- **D5_SAH1_test** (Tampere SAH1 test): 156 samples × 775 APs; sentinel=100.0 (frac=0.949); coords cols=['x', 'y', 'z', 'floor']; RSSI mean=-78.35800653594771, median=-80.0
- **D6_train** (Crowdsourced training): 697 samples × 992 APs; sentinel=100.0 (frac=0.952); coords cols=['x', 'y', 'z']; RSSI mean=-77.00609865470852, median=-80.0
- **D6_test** (Crowdsourced test): 3951 samples × 992 APs; sentinel=100.0 (frac=0.950); coords cols=['x', 'y', 'z']; RSSI mean=-76.98800621070583, median=-79.0

## Suitability for RSSI prediction
- D5 (Tampere multi-building): metre coordinates + documented +100 sentinel + 2 buildings -> **PRIMARY**.
- D1 (UJIIndoorLoc): large, documented +100, geographic coords (need conversion) -> secondary.
- D4 (Tampere WLAN RSS): smaller single-building fingerprint -> secondary.
- D6 (Crowdsourced): heterogeneous crowdsourcing -> localization/propagation reference only.

Figures are in `analysis/figures/`.
