"""Print headers + head of coordinate files to fix coord-column detection."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def find(start, name):
    m = list(Path(start).rglob(name))
    return m[0] if m else None


targets = [
    ("D5 TIE1 train coords", ROOT / "datasets" / "extracted" / "D5", "TIE1_training_coordinates.csv"),
    ("D5 SAH1 train coords", ROOT / "datasets" / "extracted" / "D5", "SAH1_training_coordinates.csv"),
    ("D4 train coords", ROOT / "datasets" / "raw" / "zenodo" / "tampere_wlan_rss_1161525", "Training_coordinates.csv"),
    ("D6 train coords", ROOT / "datasets" / "extracted" / "D6", "Training_coordinates_21Aug17.csv"),
]
for label, start, name in targets:
    p = find(start, name)
    print(f"=== {label} ===")
    if p is None:
        print("  not found")
        continue
    df = pd.read_csv(p)
    print("  columns:", list(df.columns))
    print("  dtypes:", {c: str(df[c].dtype) for c in df.columns})
    print("  head:")
    print(df.head(3).to_string().replace("\n", "\n  "))
