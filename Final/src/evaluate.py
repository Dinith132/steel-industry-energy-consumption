"""Evaluate models; report metrics in scaled space and kWh."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pickle
import numpy as np
import pandas as pd

from .metrics import compute_metrics
from .paths import get_output_root, plots_dir, results_dir
from .train import load_model_for_window
from .windows import get_window_datasets


def _inverse_target(values: np.ndarray, scaler, target_idx: int, transform: str) -> np.ndarray:
    """Inverse MinMax for target column only."""
    n_features = scaler.n_features_in_
    dummy = np.zeros((len(values), n_features), dtype=np.float64)
    dummy[:, target_idx] = values.ravel()
    inv = scaler.inverse_transform(dummy)[:, target_idx]
    if transform == "log1p":
        inv = np.expm1(inv)
    return inv


def evaluate_one(
    cfg: dict[str, Any],
    model_type: str,
    window_size: int,
) -> dict[str, Any]:
    model = load_model_for_window(cfg, model_type, window_size)
    X_train, X_test, y_train, y_test, meta = get_window_datasets(cfg, window_size)

    y_pred = model.predict(X_test, verbose=0).ravel()

    scaled = compute_metrics(y_test, y_pred)

    root = get_output_root(cfg)
    with open(root / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(root / "feature_names.json", encoding="utf-8") as f:
        feat_meta = json.load(f)
    target_idx = feat_meta["feature_names"].index(feat_meta["target"])
    transform = feat_meta.get("transform", "minmax")

    y_test_kwh = _inverse_target(y_test, scaler, target_idx, transform)
    y_pred_kwh = _inverse_target(y_pred, scaler, target_idx, transform)
    kwh = compute_metrics(y_test_kwh, y_pred_kwh)

    return {
        "track": cfg["track"],
        "model_type": model_type,
        "window_size": window_size,
        "rmse_scaled": scaled["rmse"],
        "mae_scaled": scaled["mae"],
        "r2_scaled": scaled["r2"],
        "wia_scaled": scaled["wia"],
        "rmse_kwh": kwh["rmse"],
        "mae_kwh": kwh["mae"],
        "r2_kwh": kwh["r2"],
        "wia_kwh": kwh["wia"],
    }


def evaluate_all(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model_type in cfg["models"]:
        for window in cfg["window_sizes"]:
            print(f"eval {model_type} win{window}")
            try:
                rows.append(evaluate_one(cfg, model_type, window))
            except FileNotFoundError as e:
                print(f"  skip: {e}")

    df = pd.DataFrame(rows)
    out = results_dir(cfg) / f"{cfg['track']}_metrics.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    return df


def pick_best_model(cfg: dict[str, Any], metrics_df: pd.DataFrame | None = None) -> dict[str, Any]:
    if metrics_df is None:
        path = results_dir(cfg) / f"{cfg['track']}_metrics.csv"
        metrics_df = pd.read_csv(path)
    if metrics_df.empty:
        raise ValueError("No metrics available")
    best = metrics_df.loc[metrics_df["rmse_kwh"].idxmin()]
    return {
        "model_type": best["model_type"],
        "window_size": int(best["window_size"]),
        "rmse_kwh": float(best["rmse_kwh"]),
    }


def save_best_model_json(cfg: dict[str, Any], metrics_df: pd.DataFrame | None = None) -> Path:
    best = pick_best_model(cfg, metrics_df)
    path = get_output_root(cfg) / "best_model.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    return path
