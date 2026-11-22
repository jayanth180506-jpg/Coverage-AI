# Coverage Classification Evaluation

Classes derive from measured RSSI using configurable engineering thresholds. Spatial cells are held out; classification is a separate task from RSSI regression.

Thresholds (dBm): excellent >= -50; good >= -60; fair >= -70; poor >= -80; very poor < -80.

| model             |   accuracy |   macro_f1 |
|:------------------|-----------:|-----------:|
| majority_baseline |   0.35386  |   0.104549 |
| decision_tree     |   0.412935 |   0.275831 |
| random_forest     |   0.454778 |   0.331918 |

Class imbalance and public-to-ESP32 domain shift limit interpretation. Firmware can use direct configurable thresholds on measured or predicted RSSI.
