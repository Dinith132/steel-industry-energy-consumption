"""Integrated Gradients for time-step importance and memory horizon."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .paths import xai_dir
from .train import load_model_for_window
from .windows import get_window_datasets


def integrated_gradients(
    model: tf.keras.Model,
    x: np.ndarray,
    baseline: np.ndarray | None = None,
    steps: int = 50,
) -> np.ndarray:
    """
    IG attributions for one sample x shape (window, features).
    Returns attributions same shape as x.
    """
    x = tf.convert_to_tensor(x[np.newaxis, ...], dtype=tf.float32)
    if baseline is None:
        baseline = tf.zeros_like(x)
    else:
        baseline = tf.convert_to_tensor(baseline[np.newaxis, ...], dtype=tf.float32)

    alphas = tf.linspace(0.0, 1.0, steps + 1)
    grads_sum = tf.zeros_like(x)

    for alpha in alphas:
        with tf.GradientTape() as tape:
            interpolated = baseline + alpha * (x - baseline)
            tape.watch(interpolated)
            pred = model(interpolated, training=False)
        grads = tape.gradient(pred, interpolated)
        grads_sum += grads

    avg_grads = grads_sum / tf.cast(steps + 1, tf.float32)
    attributions = (x - baseline) * avg_grads
    return attributions.numpy()[0]


def memory_horizon_curve(attributions: np.ndarray) -> np.ndarray:
    """Mean absolute attribution per lag (averaged over features). Shape (window,)."""
    return np.abs(attributions).mean(axis=1)


def run_timesteps(
    cfg: dict[str, Any],
    model_type: str,
    window_size: int,
    num_samples: int = 30,
    steps: int | None = None,
) -> Path:
    model = load_model_for_window(cfg, model_type, window_size)
    X_train, X_test, _, _, _ = get_window_datasets(cfg, window_size)
    steps = steps or cfg.get("xai", {}).get("ig_steps", 50)

    baseline = X_train.mean(axis=0)
    curves = []
    for i in range(min(num_samples, len(X_test))):
        attr = integrated_gradients(model, X_test[i], baseline=baseline, steps=steps)
        curves.append(memory_horizon_curve(attr))

    mean_curve = np.mean(curves, axis=0)
    lags = np.arange(window_size)

    out_dir = xai_dir(cfg, "timesteps")
    np.save(out_dir / f"ig_horizon_{model_type}_win{window_size}.npy", mean_curve)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lags, mean_curve, marker="o")
    ax.set_xlabel("Lag (time steps back from prediction)")
    ax.set_ylabel("Mean |attribution|")
    ax.set_title(f"Memory horizon — {model_type} win{window_size}")
    fig.tight_layout()
    plot_path = out_dir / f"ig_horizon_{model_type}_win{window_size}.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path


def run_timesteps_for_windows(cfg: dict[str, Any], windows: list[int] | None = None) -> None:
    best_path = Path(cfg["_final_root"]) / "outputs" / cfg["track"] / "best_model.json"
    if best_path.exists():
        with open(best_path, encoding="utf-8") as f:
            best = json.load(f)
        model_type = best["model_type"]
    else:
        from .evaluate import pick_best_model, results_dir
        import pandas as pd

        m = pd.read_csv(results_dir(cfg) / f"{cfg['track']}_metrics.csv")
        best = pick_best_model(cfg, m)
        model_type = best["model_type"]

    if windows is None:
        windows = cfg.get("xai", {}).get("sample_windows", [8, 24, 168])
        windows = list(set(windows + [int(best["window_size"])]))

    for w in windows:
        print(f"IG timesteps win{w}")
        run_timesteps(cfg, model_type, w)
