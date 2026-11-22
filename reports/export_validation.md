# COVERAGE-AI — Python vs C++ Validation (Stage 14)

## Method
The exported Decision Tree (`wifi_rssi_model.h`) was compiled with g++ into a test harness that re-predicts the same representative inputs scikit-learn predicted in Python. The C++ traversal uses the same float64 thresholds and leaf values, so predictions must match to within floating rounding.

## Compiler
`g++ -O2 -std=c++11`

## Results

- Rows compared: 2000
- Max absolute difference: 0.000e+00 dBm
- Mean absolute difference: 0.000e+00 dBm
- Tolerance: 1e-04 dBm
- Result: **PASS**

## Conclusion
The C++ inference reproduces the Python (scikit-learn) predictions within tolerance. The exported header is safe to merge into the ESP32 Arduino sketch.
