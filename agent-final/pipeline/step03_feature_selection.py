"""
Step 03 — Apply feature selection from step02, engineer time features, preprocess.

Drops (data quality + leakage):
  - CO2(tCO2)           — target leakage (r=0.99)
  - Leading_Current_Reactive_Power_kVarh — 67% zeros, redundant with Leading PF
  - NSM                 — r=0.999 with hour (pure time counter)

Adds:
  - hour_sin, hour_cos  — cyclical time (replaces NSM properly)

Run AFTER step02: python pipeline/step03_feature_selection.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "step03"
STEP02 = ROOT / "outputs" / "step02"
RAW = ROOT.parent / "Steel_industry_data.csv"
TARGET = "Usage_kWh"
TEST_RATIO = 0.18

DROP = ["CO2(tCO2)", "Leading_Current_Reactive_Power_kVarh", "NSM"]
CATEGORICAL = ["WeekStatus", "Day_of_week", "Load_Type"]


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
    df = df.sort_values("date").reset_index(drop=True)

    # Engineered cyclical hour (replaces NSM)
    hour = df["date"].dt.hour + df["date"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df = df.drop(columns=[c for c in DROP if c in df.columns])
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    mapping = {}
    for col in CATEGORICAL:
        enc = OrdinalEncoder()
        out[col] = enc.fit_transform(out[[col]].astype(str)).ravel()
        mapping[col] = {str(c): int(i) for i, c in enumerate(enc.categories_[0])}
    return out, mapping


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    d = df.set_index("date")
    num_cols = [c for c in d.columns if c not in CATEGORICAL + ["hour_sin", "hour_cos"]]
    agg = {c: "mean" for c in num_cols if c != TARGET}
    agg[TARGET] = "mean"
    agg["hour_sin"] = "mean"
    agg["hour_cos"] = "mean"
    for c in CATEGORICAL:
        agg[c] = "first"
    return d.resample("h").agg(agg).reset_index()


def save_track(df: pd.DataFrame, track: str) -> dict:
    feature_cols = [c for c in df.columns if c != "date"]
    ordered = [TARGET] + [c for c in feature_cols if c != TARGET]
    df = df[["date"] + ordered]

    n = len(df)
    split_idx = int(n * (1 - TEST_RATIO))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    scaler = MinMaxScaler()
    scaler.fit(train_df[ordered])
    train_s = train_df.copy()
    test_s = test_df.copy()
    train_s[ordered] = scaler.transform(train_df[ordered])
    test_s[ordered] = scaler.transform(test_df[ordered])
    full = pd.concat([train_s, test_s], ignore_index=True)

    track_out = OUT / track
    track_out.mkdir(parents=True, exist_ok=True)
    full.drop(columns=["date"]).to_csv(track_out / "data.csv", index=False)
    with open(track_out / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "track": track,
        "target": TARGET,
        "dropped_features": DROP,
        "engineered": ["hour_sin", "hour_cos"],
        "feature_names": ordered,
        "n_features": len(ordered),
        "split_idx": split_idx,
        "n_train": split_idx,
        "n_test": n - split_idx,
        "n_rows": n,
    }
    (track_out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def validate_clean(df: pd.DataFrame, ordered: list[str]) -> dict:
    """Post-clean correlation check — no feature should have |r|>0.95 with target except Usage lag in windows."""
    num = [c for c in ordered if c != TARGET]
    corr = df[num + [TARGET]].corr()[TARGET].drop(TARGET, errors="ignore")
    if isinstance(corr, pd.DataFrame):
        corr = corr.iloc[:, 0]
    leakage = {k: round(float(v), 4) for k, v in corr.items() if abs(v) >= 0.95}
    return {
        "target_correlations": {k: round(float(v), 4) for k, v in corr.items()},
        "remaining_leakage_flags": leakage,
        "pass": len(leakage) == 0,
    }


def main():
    print("=" * 60)
    print("STEP 03 — Feature selection & preprocess")
    print("=" * 60)

    step02 = json.loads((STEP02 / "step02_report.json").read_text())
    drop_list = step02["decisions"]["proposed_drop"]
    print(f"Dropping: {drop_list}")

    df = load_and_clean()
    df, cat_map = encode_categoricals(df)

    # Validate on 15min before scaling
    ordered_pre = [TARGET] + [c for c in df.columns if c not in ("date", TARGET)]
    validation = validate_clean(df, ordered_pre)

    meta_15 = save_track(df, "15min")
    hourly = to_hourly(df)
    meta_h = save_track(hourly, "hourly")

    # Plot cleaned correlation
    OUT.mkdir(parents=True, exist_ok=True)
    corr = df[ordered_pre].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 8})
    ax.set_title("Step 03 — Correlation AFTER feature selection (15min)")
    fig.tight_layout()
    fig.savefig(OUT / "01_clean_correlation.png", dpi=130)
    plt.close(fig)

    feature_manifest = {
        "dropped": {f: step02["decisions"].get("zero_action", {}).get(f, step02["decisions"]["leakage_action"].get(f, "redundant")) for f in DROP},
        "kept": meta_15["feature_names"],
        "engineered": {"hour_sin": "cyclical hour", "hour_cos": "cyclical hour"},
        "categorical_encoding": cat_map,
        "validation": validation,
        "tracks": {"15min": meta_15, "hourly": meta_h},
        "next_step": "step04_windows — only after validation pass",
    }
    (OUT / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Step 03 — Clean feature set",
        "",
        f"**Validation pass:** {validation['pass']}",
        "",
        "## Dropped features",
        "| Feature | Reason |",
        "|---------|--------|",
    ]
    for f in DROP:
        reason = feature_manifest["dropped"].get(f, "see step02")
        lines.append(f"| {f} | {reason} |")
    lines += [
        "",
        "## Final features (10 → 7 inputs + target)",
        f"```\n{meta_15['feature_names']}\n```",
        "",
        "## Target correlation after cleaning",
        "```json",
        json.dumps(validation["target_correlations"], indent=2),
        "```",
        "",
        f"## Tracks: 15min {meta_15['n_rows']} rows, hourly {meta_h['n_rows']} rows",
        "",
        "## Next",
        "**step04_windows**" if validation["pass"] else "**FIX leakage before continuing**",
    ]
    (OUT / "DECISIONS.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Features: {meta_15['feature_names']}")
    print(f"Validation pass: {validation['pass']}")
    if validation["remaining_leakage_flags"]:
        print(f"WARNING leakage: {validation['remaining_leakage_flags']}")
    print(f"15min: {meta_15['n_rows']} rows | hourly: {meta_h['n_rows']} rows")
    print(f"Wrote {OUT}/DECISIONS.md")


if __name__ == "__main__":
    main()
