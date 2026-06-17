"""
Step 03 — Apply step02 drops, engineer time features, save 10min + hourly tracks.

Run AFTER step02: python pipeline/step03_preprocess.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "step03"
STEP02 = ROOT / "outputs" / "step02"
RAW = ROOT.parent / "energydata_complete.csv"
TARGET = "Appliances"
TEST_RATIO = 0.18

SUM_COLS = ["Appliances", "lights"]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW, parse_dates=["date"])
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if col == "date":
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hour = out["date"].dt.hour + out["date"].dt.minute / 60.0
    dow = out["date"].dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return out


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    d = df.set_index("date")
    agg = {}
    for col in d.columns:
        if col in SUM_COLS:
            agg[col] = "sum"
        else:
            agg[col] = "mean"
    return d.resample("h").agg(agg).reset_index()


def save_track(df: pd.DataFrame, track: str, drop_list: list[str]) -> dict:
    work = df.drop(columns=[c for c in drop_list if c in df.columns])
    feature_cols = [c for c in work.columns if c != "date"]
    ordered = [TARGET] + [c for c in feature_cols if c != TARGET]
    work = work[["date"] + ordered]

    n = len(work)
    split_idx = int(n * (1 - TEST_RATIO))
    train_df = work.iloc[:split_idx]
    test_df = work.iloc[split_idx:]

    scaler = MinMaxScaler()
    scaler.fit(train_df[ordered])
    train_s = train_df.copy()
    test_s = test_df.copy()
    train_s[ordered] = scaler.transform(train_df[ordered])
    test_s[ordered] = scaler.transform(test_df[ordered])
    full = pd.concat([train_s, test_s], ignore_index=True)

    prep_out = ROOT / "outputs" / "preprocess" / track
    prep_out.mkdir(parents=True, exist_ok=True)
    full.drop(columns=["date"]).to_csv(prep_out / "data.csv", index=False)
    with open(prep_out / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "track": track,
        "target": TARGET,
        "dropped_features": drop_list,
        "feature_names": ordered,
        "n_features": len(ordered),
        "split_idx": split_idx,
        "n_train": split_idx,
        "n_test": n - split_idx,
        "n_rows": n,
    }
    (prep_out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT / track).mkdir(parents=True, exist_ok=True)
    (OUT / track / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def validate_clean(df: pd.DataFrame, ordered: list[str]) -> dict:
    num = [c for c in ordered if c != TARGET]
    corr = df[num + [TARGET]].corr()[TARGET].drop(TARGET, errors="ignore")
    leakage = {k: round(float(v), 4) for k, v in corr.items() if abs(v) >= 0.95}
    return {
        "target_correlations": {k: round(float(v), 4) for k, v in corr.items()},
        "remaining_leakage_flags": leakage,
        "pass": len(leakage) == 0,
    }


def main():
    print("=" * 60)
    print("STEP 03 — Preprocess")
    print("=" * 60)

    step02 = json.loads((STEP02 / "step02_report.json").read_text())
    drop_list = step02["decisions"]["proposed_drop"]
    print(f"Dropping: {drop_list}")

    df = load_raw()
    df = add_time_features(df)

    # validate on raw-scale 10min before scaling
    work = df.drop(columns=[c for c in drop_list if c in df.columns])
    ordered_pre = [TARGET] + [c for c in work.columns if c not in ("date", TARGET)]
    validation = validate_clean(work, ordered_pre)

    meta_10 = save_track(df, "10min", drop_list)
    hourly = to_hourly(df)
    meta_h = save_track(hourly, "hourly", drop_list)

    OUT.mkdir(parents=True, exist_ok=True)
    corr = work[ordered_pre].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation after feature selection (10min, unscaled)")
    fig.tight_layout()
    fig.savefig(OUT / "01_clean_correlation.png", dpi=120)
    plt.close(fig)

    manifest = {
        "dropped": step02["decisions"]["drop_reason"],
        "kept": meta_10["feature_names"],
        "validation": validation,
        "tracks": {"10min": meta_10, "hourly": meta_h},
    }
    (OUT / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Step 03 — Clean feature set",
        "",
        f"**Validation pass:** {validation['pass']}",
        "",
        "## Features",
        f"```\n{meta_10['feature_names']}\n```",
        "",
        f"## Tracks: 10min {meta_10['n_rows']} rows | hourly {meta_h['n_rows']} rows",
        "",
        "## Next",
        "**step04_windows**",
    ]
    (OUT / "DECISIONS.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Features ({len(meta_10['feature_names'])}): {meta_10['feature_names']}")
    print(f"Validation pass: {validation['pass']}")
    print(f"10min: {meta_10['n_rows']} | hourly: {meta_h['n_rows']}")
    print(f"Wrote {OUT}/DECISIONS.md")


if __name__ == "__main__":
    main()
