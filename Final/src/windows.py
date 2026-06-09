"""Sliding-window construction for LSTM sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_scaled_matrix(cfg: dict[str, Any]) -> tuple[np.ndarray, list[str], dict]:
    from .paths import get_output_root

    root = get_output_root(cfg)
    with open(root / "feature_names.json", encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_csv(root / "data.csv")
    names = meta["feature_names"]
    target = meta["target"]
    target_idx = names.index(target)
    data = df[names].to_numpy(dtype=np.float32)
    return data, names, meta


def create_sliding_windows(
    data: np.ndarray,
    window_size: int,
    target_idx: int = 0,
    include_target_lags: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(data) - window_size):
        window = data[i : i + window_size]
        if include_target_lags:
            X.append(window)
        else:
            # Legacy: exclude target column from features
            mask = np.ones(window.shape[1], dtype=bool)
            mask[target_idx] = False
            X.append(window[:, mask])
        y.append(data[i + window_size, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_test_window_split(
    data: np.ndarray,
    window_size: int,
    split_idx: int,
    target_idx: int = 0,
    include_target_lags: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split by time on raw series index, then build windows."""
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    X_train, y_train = create_sliding_windows(
        train_data, window_size, target_idx, include_target_lags
    )
    X_test, y_test = create_sliding_windows(
        test_data, window_size, target_idx, include_target_lags
    )
    return X_train, X_test, y_train, y_test


def get_window_datasets(cfg: dict[str, Any], window_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    data, names, meta = load_scaled_matrix(cfg)
    target_idx = names.index(meta["target"])
    include_lags = meta.get("include_target_lags", cfg.get("include_target_lags", True))
    split_idx = meta["split_idx"]
    X_train, X_test, y_train, y_test = train_test_window_split(
        data, window_size, split_idx, target_idx, include_lags
    )
    return X_train, X_test, y_train, y_test, meta
