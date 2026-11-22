# Indoor Localization Evaluation (separate objective)

Model: Random Forest regressor maps the 520 UJI WAP readings to X/Y. Missing APs are represented by a separate detection mask; +100 is not used as RSSI. Whole 5x5 spatial cells per building were held out.

## Spatial holdout metrics

- n_train: 14782
- n_test: 5155
- mae_x: 18.720144194913352
- mae_y: 12.975056134346262
- rmse_x: 24.93713140762481
- rmse_y: 18.24277545505167
- median_euclidean_error_coordinate_units: 17.69452668883699
- mean_euclidean_error_coordinate_units: 24.118265927055187

UJI coordinate columns are projected planar values, not GPS degrees. The exact source CRS/origin is not established, so errors are reported in coordinate units; these results do not transfer directly to the ESP32 local frame.
