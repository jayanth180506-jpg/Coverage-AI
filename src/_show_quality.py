import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "analysis" / "dataset_quality_report.csv")
for _, r in df.iterrows():
    print(f"{r.tag:16} samples={str(r.n_samples):>6} aps={str(r.n_aps):>4} "
          f"sent={r.sentinel} mean={round(r.rssi_mean,2):>7} "
          f"x={r.lon_or_x} | y={r.lat_or_y} | floor={r.floor}")
