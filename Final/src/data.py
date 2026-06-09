"""Load, resample, encode, split, and scale time-series data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from .paths import get_output_root, resolve_data_csv


NUMERIC_COLS_DEFAULT = [
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
]


def load_raw_dataframe(cfg: dict[str, Any]) -> pd.DataFrame:
    path = resolve_data_csv(cfg)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _encode_categoricals(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    cols = [c for c in cfg["preprocessing"]["ordinal_encode"] if c in df.columns]
    if not cols:
        return df
    out = df.copy()
    # Already numeric in preprocessed v1
    for c in cols:
        if out[c].dtype == object or str(out[c].dtype) == "string":
            enc = OrdinalEncoder()
            out[c] = enc.fit_transform(out[[c]])
    return out


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("date")
    agg = {
        "Usage_kWh": "mean",
        "Lagging_Current_Reactive.Power_kVarh": "mean",
        "Leading_Current_Reactive_Power_kVarh": "mean",
        "CO2(tCO2)": "mean",
        "Lagging_Current_Power_Factor": "mean",
        "Leading_Current_Power_Factor": "mean",
        "NSM": "mean",
        "WeekStatus": "first",
        "Load_Type": "first",
    }
    agg = {k: v for k, v in agg.items() if k in df.columns}
    return df.resample("h").agg(agg).reset_index()


def preprocess(cfg: dict[str, Any]) -> dict[str, Any]:
    df = load_raw_dataframe(cfg)
    df = _encode_categoricals(df, cfg)

    if cfg.get("resample", False):
        df = resample_hourly(df)

    if cfg.get("remove_co2") and "CO2(tCO2)" in df.columns:
        df = df.drop(columns=["CO2(tCO2)"])

    feature_cols = [c for c in df.columns if c != "date"]
    target = cfg["target"]
    if target not in feature_cols:
        raise ValueError(f"Target {target} not in columns")

    # Reorder: target first (for sliding windows)
    other = [c for c in feature_cols if c != target]
    ordered = [target] + other
    df = df[["date"] + ordered]

    test_ratio = cfg["preprocessing"]["test_ratio"]
    n = len(df)
    split_idx = int(n * (1 - test_ratio))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    numeric_cols = ordered

    transform = cfg["preprocessing"].get("transform", "minmax")
    if transform == "log1p":
        train_df[numeric_cols] = np.log1p(train_df[numeric_cols].astype(float))
        test_df[numeric_cols] = np.log1p(test_df[numeric_cols].astype(float))

    scaler = MinMaxScaler()
    scaler.fit(train_df[numeric_cols])
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[numeric_cols] = scaler.transform(train_df[numeric_cols])
    test_scaled[numeric_cols] = scaler.transform(test_df[numeric_cols])

    full_scaled = pd.concat([train_scaled, test_scaled], ignore_index=True)

    out_root = get_output_root(cfg)
    out_root.mkdir(parents=True, exist_ok=True)

    full_scaled.drop(columns=["date"]).to_csv(out_root / "data.csv", index=False)
    with open(out_root / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "feature_names": ordered,
        "target": target,
        "split_idx": split_idx,
        "n_train": split_idx,
        "n_test": n - split_idx,
        "n_rows": n,
        "track": cfg["track"],
        "include_target_lags": cfg.get("include_target_lags", True),
        "transform": transform,
    }
    with open(out_root / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(out_root / "split_indices.json", "w", encoding="utf-8") as f:
        json.dump({"split_idx": split_idx, "n_rows": n}, f, indent=2)

    return meta
