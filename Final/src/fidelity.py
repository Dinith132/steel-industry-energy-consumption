"""
Phase 2: Feature/time perturbation fidelity tests.

Run after best_model.json exists:
  python -m src.run fidelity --track hourly
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .evaluate import evaluate_one
from .metrics import compute_metrics
from .paths import get_output_root, results_dir
from .train import load_model_for_window
from .windows import get_window_datasets


def perturb_top_feature(
    cfg: dict[str, Any],
    model_type: str,
    window_size: int,
    top_feature_idx: int = 1,
) -> dict[str, float]:
    """Zero out one feature channel across the window; measure RMSE change."""
    model = load_model_for_window(cfg, model_type, window_size)
    _, X_test, _, y_test, _ = get_window_datasets(cfg, window_size)

    y_pred_base = model.predict(X_test, verbose=0).ravel()
    base_rmse = compute_metrics(y_test, y_pred_base)["rmse"]

    X_pert = X_test.copy()
    X_pert[:, :, top_feature_idx] = 0.0
    y_pred_pert = model.predict(X_pert, verbose=0).ravel()
    pert_rmse = compute_metrics(y_test, y_pred_pert)["rmse"]

    return {
        "baseline_rmse_scaled": base_rmse,
        "perturbed_rmse_scaled": pert_rmse,
        "rmse_increase": pert_rmse - base_rmse,
        "feature_idx": top_feature_idx,
    }


def run_fidelity(cfg: dict[str, Any]) -> dict[str, float]:
    best_path = get_output_root(cfg) / "best_model.json"
    with open(best_path, encoding="utf-8") as f:
        best = json.load(f)
    result = perturb_top_feature(
        cfg, best["model_type"], int(best["window_size"]), top_feature_idx=1
    )
    out = results_dir(cfg) / "fidelity_perturbation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
