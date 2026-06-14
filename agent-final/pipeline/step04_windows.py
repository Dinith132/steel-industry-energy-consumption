"""Step 04 — Window validation on CLEAN data (step03 outputs)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "step04"
DATA = ROOT / "outputs" / "step03"
TARGET = "Usage_kWh"
TEST_RATIO = 0.18

WINDOWS = {
    "hourly": [1, 4, 8, 12, 16, 24, 36, 48, 74, 168, 336, 672],
    "15min": [1, 4, 8, 16, 24, 48, 64, 96, 672],
}


def build_windows(arr, window, target_idx=0):
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i : i + window])
        y.append(arr[i + window, target_idx])
    return np.array(X, np.float32), np.array(y, np.float32)


def main():
    print("=" * 60)
    print("STEP 04 — Window validation (clean features)")
    print("=" * 60)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}

    for track, wins in WINDOWS.items():
        meta = json.loads((DATA / track / "meta.json").read_text())
        cols = meta["feature_names"]
        arr = pd.read_csv(DATA / track / "data.csv")[cols].to_numpy(np.float32)
        tidx = cols.index(TARGET)
        report[track] = {}
        for w in wins:
            X, _ = build_windows(arr, w, tidx)
            split = int(len(X) * (1 - TEST_RATIO))
            report[track][str(w)] = {"n_samples": len(X), "n_train": split, "X_shape": [len(X), w, len(cols)]}
            print(f"  {track} win{w:>3}: {len(X)} samples, shape ({w}, {len(cols)} features)")

    report["recommended"] = {"hourly": 24, "15min": 96, "reason": "EDA daily cycle; prior study sweet spot"}
    report["next_step"] = "step05_train — Colab + TensorFlow on clean 9-feature set"
    (OUT / "windows_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nClean feature count per timestep: {len(cols)}")
    print("Next: step05_train (Colab)")


if __name__ == "__main__":
    main()
