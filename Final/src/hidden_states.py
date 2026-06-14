"""
Phase 2: Extract LSTM hidden states and visualize with PCA/t-SNE.

Run after training:
  python -m src.run hidden_states --track hourly
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

from .paths import get_output_root, plots_dir
from .train import load_model_for_window
from .windows import get_window_datasets


def extract_hidden_states(
    model: tf.keras.Model,
    X: np.ndarray,
    layer_index: int = 0,
) -> np.ndarray:
    """Last hidden state per sample from first LSTM layer."""
    _ = model.predict(X[:1], verbose=0)
    layer = model.layers[layer_index]
    sub = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
    out = sub.predict(X, verbose=0)
    if out.ndim == 3:
        return out[:, -1, :]
    return out


def run_hidden_state_pca(cfg: dict[str, Any], max_samples: int = 500) -> None:
    best_path = get_output_root(cfg) / "best_model.json"
    with open(best_path, encoding="utf-8") as f:
        best = json.load(f)

    model = load_model_for_window(cfg, best["model_type"], int(best["window_size"]))
    X_train, _, _, _, _ = get_window_datasets(cfg, int(best["window_size"]))
    states = extract_hidden_states(model, X_train[:max_samples])

    pca = PCA(n_components=2)
    coords = pca.fit_transform(states)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.5, s=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Hidden states PCA — {best['model_type']} win{best['window_size']}")
    out = plots_dir(cfg, "phase2") / "hidden_states_pca.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")
