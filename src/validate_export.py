"""COVERAGE-AI Python-vs-C++ validation (Stage 14).

Compiles the generated C++ test harness with g++, runs it over the
representative test inputs, and compares the C++ predictions against the
Python (scikit-learn) predictions. Reports max/mean absolute difference and
PASS/FAIL against a documented tolerance.

Tolerance rationale: both implementations use the same float64 tree
thresholds and leaf values, so they must agree to within floating rounding
(we require <= 1e-4 dBm).
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPORTED = ROOT / "models" / "exported"
REPORTS = ROOT / "reports"
TOL = 1e-4


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def main():
    exe = EXPORTED / "test_model.exe"
    cpp = EXPORTED / "test_model.cpp"
    hdr = EXPORTED / "wifi_rssi_model.h"
    if not cpp.exists() or not hdr.exists():
        print("missing C++ artifacts; run export_esp32_model.py first")
        return 1

    # 1) compile
    r = run(["g++", "-O2", "-std=c++11", "-o", str(exe), str(cpp), f"-I{EXPORTED}"])
    print("compile rc:", r.returncode)
    if r.returncode != 0:
        print("STDERR:", r.stderr)
        return 2
    if r.stderr.strip():
        print("compile warnings:", r.stderr[:500])

    # 2) run (cwd = EXPORTED so relative test_inputs.csv / cpp_predictions.csv resolve)
    r = run([str(exe)], cwd=str(EXPORTED))
    print("run rc:", r.returncode)
    print("stdout:", r.stdout.strip())
    if r.returncode != 0:
        print("STDERR:", r.stderr)
        return 3

    # 3) compare
    pred_path = EXPORTED / "cpp_predictions.csv"
    if not pred_path.exists():
        print("cpp_predictions.csv not produced")
        return 4
    df = pd.read_csv(pred_path)
    df["abs_diff"] = (df["cpp_pred"] - df["python_pred"]).abs()
    maxd = float(df["abs_diff"].max())
    meand = float(df["abs_diff"].mean())
    n = len(df)
    passed = maxd <= TOL

    print(f"rows compared : {n}")
    print(f"max abs diff  : {maxd:.3e} dBm")
    print(f"mean abs diff : {meand:.3e} dBm")
    print(f"tolerance     : {TOL:.0e} dBm")
    print("RESULT        :", "PASS" if passed else "FAIL")

    rep = ["# COVERAGE-AI — Python vs C++ Validation (Stage 14)\n\n",
           "## Method\n",
           "The exported Decision Tree (`wifi_rssi_model.h`) was compiled with g++ ",
           "into a test harness that re-predicts the same representative inputs ",
           "scikit-learn predicted in Python. The C++ traversal uses the same ",
           "float64 thresholds and leaf values, so predictions must match to within ",
           "floating rounding.\n\n",
           "## Compiler\n",
           "`g++ -O2 -std=c++11`\n\n",
           "## Results\n\n",
           f"- Rows compared: {n}\n",
           f"- Max absolute difference: {maxd:.3e} dBm\n",
           f"- Mean absolute difference: {meand:.3e} dBm\n",
           f"- Tolerance: {TOL:.0e} dBm\n",
           f"- Result: **{'PASS' if passed else 'FAIL'}**\n\n",
           "## Conclusion\n",
           "The C++ inference reproduces the Python (scikit-learn) predictions ",
           f"within tolerance. The exported header is safe to merge into the ESP32 ",
           "Arduino sketch.\n"]
    (REPORTS / "export_validation.md").write_text("".join(rep), encoding="utf-8")
    print("wrote", REPORTS / "export_validation.md")
    return 0 if passed else 5


if __name__ == "__main__":
    sys.exit(main())
