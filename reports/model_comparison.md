# COVERAGE-AI — Model Comparison Report (Stage 11)
## Held-out spatial test set
The test set is made of whole spatial grid cells held out via GroupShuffleSplit (no nearby correlated points leak into the test set).

## Metrics

| model                  |     MAE |     RMSE |           R2 |   fit_seconds |   inference_us_per_row |   n_features |      model_bytes |
|:-----------------------|--------:|---------:|-------------:|--------------:|-----------------------:|-------------:|-----------------:|
| global_mean_baseline   | 7.87938 |  9.69817 |  -0.00287396 |   nan         |            nan         |          nan |    nan           |
| global_median_baseline | 7.47862 |  9.86595 |  -0.0378738  |   nan         |            nan         |          nan |    nan           |
| per_ap_mean_baseline   | 7.76592 |  9.87342 |  -0.0394479  |   nan         |            nan         |          nan |    nan           |
| per_ap_median_baseline | 7.56406 | 10.1953  |  -0.10832    |   nan         |            nan         |          nan |    nan           |
| decision_tree          | 7.82097 |  9.95865 |  -0.0574695  |   nan         |            nan         |            5 |   3381           |
| linear_regression      | 7.73213 |  9.84177 |  -0.032793   |     0.235717  |              0.139364  |            5 |    646           |
| ridge_regression       | 7.73213 |  9.84177 |  -0.0327929  |     0.0685566 |              0.0388969 |            5 |    626           |
| random_forest          | 6.28244 |  8.5414  |   0.222098   |    17.0901    |              0.873648  |            5 |      8.94938e+07 |
| gradient_boosting      | 7.40451 |  9.5336  |   0.0308733  |    70.8519    |              1.18419   |            5 | 134650           |
| per_ap_tree_mean       | 8.38852 | 10.3617  | nan          |   nan         |            nan         |          nan |    nan           |

## Per-building performance

| building_id   |     n |     MAE |     RMSE |
|:--------------|------:|--------:|---------:|
| SAH1          | 61421 | 8.3631  | 10.4658  |
| TIE1          | 59460 | 7.26097 |  9.40607 |

## Notes
- Final Decision Tree: max_depth=4, min_samples_leaf=50, CV-MAE=7.780.
- Feature importances are recorded in `analysis/feature_analysis.csv`; interpret them as model-specific and limited to this dataset and split.
- The Decision Tree essentially matches the per-AP median baseline (MAE ~7.7-7.8 dBm). This is an honest result: RSSI at a given location for a given AP is dominated by the AP's average level; spatial prediction to *unseen* areas is hard with public data.

## Model selection
The full candidate set was evaluated on the same spatial test split. Random Forest has the lowest test MAE (6.28 dBm) and RMSE (8.54 dBm), but its serialized estimator is 89.5 MB and is not an ESP32 candidate. Ridge is the smallest learned candidate (626 bytes, MAE 7.73 dBm); its 1,319-entry AP table is included in the generated header. Python/C++ parity passed within 1.4e-14 dBm. The Arduino sketch uses the Ridge header. The actual board build and RAM/flash usage still require validation on the TTGO T-Call.

Figures: models/comparison/*.png
