"""Step 04 — Window validation on step03 outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "step04"
DATA = ROOT / "outputs" / "preprocess"
TARGET = "Appliances"
TEST_RATIO = 0.18

WINDOWS = {
    "10min": [1, 6, 12, 36, 72, 144, 288],
    "hourly": [1, 4, 8, 12, 24, 48, 72, 168],
}


def build_windows(arr, window, target_idx=0):
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i : i + window])
        y.append(arr[i + window, target_idx])
    return np.array(X, np.float32), np.array(y, np.float32)


def main():
    print("=" * 60)
    print("STEP 04 — Window validation")
    print("=" * 60)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"windows": WINDOWS, "tracks": {}}

    for track, wins in WINDOWS.items():
        meta = json.loads((DATA / track / "meta.json").read_text())
        cols = meta["feature_names"]
        arr = pd.read_csv(DATA / track / "data.csv")[cols].to_numpy(np.float32)
        tidx = cols.index(TARGET)
        report["tracks"][track] = {}
        for w in wins:
            X, _ = build_windows(arr, w, tidx)
            split = int(len(X) * (1 - TEST_RATIO))
            ok = split > 100 and (len(X) - split) > 50
            report["tracks"][track][str(w)] = {
                "n_samples": len(X),
                "n_train": split,
                "n_test": len(X) - split,
                "X_shape": [len(X), w, len(cols)],
                "ok": ok,
            }
            status = "OK" if ok else "LOW"
            print(f"  {track} win{w:>3}: {len(X)} samples [{status}]")

    report["recommended"] = {"10min": 36, "hourly": 24, "reason": "balance context vs sample count"}
    report["next_step"] = "train notebooks — single/double/bidir LSTM"
    (OUT / "windows_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}/windows_report.json")


if __name__ == "__main__":
    main()
