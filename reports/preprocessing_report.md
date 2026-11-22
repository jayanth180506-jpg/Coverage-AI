# Preprocessing Report

The +100 dBm source sentinel is dropped as undetected AP data. Unavailable common-schema fields remain missing. Source train/validation/test files remain separate; D5_long.csv and D4_long.csv are convenience training views only.

## Generated files
- `D1_long.csv`: 358763 detected readings; 465 AP identifiers; RSSI -104.0 to 0.0 dBm.
- `D1_training_long.csv`: 358763 detected readings; 465 AP identifiers; RSSI -104.0 to 0.0 dBm.
- `D1_validation_long.csv`: 18304 detected readings; 367 AP identifiers; RSSI -102.0 to -34.0 dBm.
- `D4_long.csv`: 18973 detected readings; 489 AP identifiers; RSSI -93.0 to -23.0 dBm.
- `D4_test_long.csv`: 34179 detected readings; 296 AP identifiers; RSSI -93.0 to -22.0 dBm.
- `D4_training_long.csv`: 18973 detected readings; 489 AP identifiers; RSSI -93.0 to -23.0 dBm.
- `D5_long.csv`: 752465 detected readings; 1364 AP identifiers; RSSI -94.0 to -27.0 dBm.
- `D5_test_long.csv`: 9780 detected readings; 309 AP identifiers; RSSI -94.0 to -43.0 dBm.
- `D5_training_long.csv`: 752465 detected readings; 1364 AP identifiers; RSSI -94.0 to -27.0 dBm.
- `D6_test_long.csv`: 196435 detected readings; 989 AP identifiers; RSSI -102.0 to -16.0 dBm.
- `D6_training_long.csv`: 33450 detected readings; 779 AP identifiers; RSSI -99.0 to -14.0 dBm.

D6 source split names are preserved because its bundled README conflicts on their train/test semantics. D1 coordinates are projected planar values; exact CRS/origin remains unknown. Public-device RSSI requires ESP32-specific calibration.
