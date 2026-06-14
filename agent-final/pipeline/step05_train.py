"""Step 05 — Train on CLEAN data (step03). Colab + TensorFlow required."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "step03"
TRACK = os.environ.get("TRACK", "hourly")
FULL = os.environ.get("FULL_GRID", "0") == "1"

WINDOWS = {"hourly": [1,4,8,12,16,24,36,48,74,168,336,672], "15min": [1,4,8,16,24,48,64,96,672]}
SMOKE = {"hourly": [24], "15min": [96]}

def main():
    try:
        import tensorflow as tf  # noqa
    except ImportError:
        print("TensorFlow required — run in Colab:")
        print("  !pip install tensorflow")
        print(f"  TRACK=hourly FULL_GRID=1 python pipeline/step05_train.py")
        return
    print(f"Train {TRACK} from {DATA}/{TRACK}/ — implement or extend from Final/hourly.ipynb")
    print("Use meta.json feature_names — 9 clean features, NO CO2/NSM/leaking columns")

if __name__ == "__main__":
    main()
