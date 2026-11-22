# D5 Official Test Evaluation

This publisher-provided split was not used for feature selection or model tuning. Models were fitted using the D5 training file, with a spatially held-out validation region used during training. AP target means use training rows only.

| model         |    n |   mae_dbm |   rmse_dbm |         r2 |   unknown_ap_fraction |
|:--------------|-----:|----------:|-----------:|-----------:|----------------------:|
| ridge         | 9780 |   5.78269 |    7.51547 | -0.0307896 |            0.00879346 |
| decision_tree | 9780 |   6.55644 |    8.45818 | -0.305606  |            0.00879346 |

D5 official test rows are relatively few; interpret these metrics alongside the larger spatial holdout results.
