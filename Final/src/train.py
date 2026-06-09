"""Train LSTM models for all window sizes and architectures."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from .models import build_lstm_model, compile_model, load_trained_model
from .paths import history_dir, model_dir
from .windows import get_window_datasets


def _model_path(cfg: dict[str, Any], model_type: str, window: int) -> Path:
    return model_dir(cfg, model_type) / f"win{window}.keras"


def _history_path(cfg: dict[str, Any], model_type: str, window: int) -> Path:
    return history_dir(cfg, model_type) / f"win{window}.pkl"


def train_one(
    cfg: dict[str, Any],
    model_type: str,
    window_size: int,
    force: bool = False,
) -> Path:
    path = _model_path(cfg, model_type, window_size)
    if path.exists() and not force:
        print(f"  skip (exists): {path}")
        return path

    X_train, X_test, y_train, y_test, _ = get_window_datasets(cfg, window_size)
    input_shape = (window_size, X_train.shape[2])

    model = build_lstm_model(model_type, input_shape, cfg)
    compile_model(model, cfg)

    training = cfg.get("training", {})
    history = model.fit(
        X_train,
        y_train,
        epochs=training.get("epochs", 50),
        batch_size=training.get("batch_size", 32),
        validation_data=(X_test, y_test),
        verbose=1,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    with open(_history_path(cfg, model_type, window_size), "wb") as f:
        pickle.dump(history.history, f)
    print(f"  saved: {path}")
    return path


def train_all(cfg: dict[str, Any], force: bool = False) -> None:
    for model_type in cfg["models"]:
        print(f"\n=== {model_type} ===")
        for window in cfg["window_sizes"]:
            print(f"window {window}")
            train_one(cfg, model_type, window, force=force)


def load_model_for_window(cfg: dict[str, Any], model_type: str, window: int):
    path = _model_path(cfg, model_type, window)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return load_trained_model(str(path))
