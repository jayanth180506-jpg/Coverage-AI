"""Compile Ridge inference and compare it with sklearn predictions."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "exported"
TOLERANCE = 1e-5


def main():
    exe = OUT / "ridge_test.exe"
    compile_result = subprocess.run(["g++", "-O2", "-std=c++11", "-o", str(exe),
                                     str(OUT / "ridge_test.cpp"), f"-I{OUT}"],
                                    capture_output=True, text=True)
    if compile_result.returncode:
        print(compile_result.stderr)
        return 2
    run = subprocess.run([str(exe)], cwd=OUT, capture_output=True, text=True)
    if run.returncode:
        print(run.stderr)
        return 3
    m = re.search(r"rows=(\d+) max_abs_diff=([\deE+.-]+) mean_abs_diff=([\deE+.-]+)", run.stdout)
    if not m:
        print("Unexpected validation output:", run.stdout)
        return 4
    rows, maxdiff, mean = int(m.group(1)), float(m.group(2)), float(m.group(3))
    passed = rows > 0 and maxdiff <= TOLERANCE
    report = ("# Ridge Python/C++ Export Validation\n\n"
              f"- Rows compared: {rows}\n- Maximum absolute difference: {maxdiff:.3e} dBm\n"
              f"- Mean absolute difference: {mean:.3e} dBm\n- Tolerance: {TOLERANCE:.0e} dBm\n"
              f"- Result: **{'PASS' if passed else 'FAIL'}**\n")
    (ROOT / "reports" / "ridge_export_validation.md").write_text(report, encoding="utf-8")
    print(report)
    return 0 if passed else 5


if __name__ == "__main__":
    sys.exit(main())
