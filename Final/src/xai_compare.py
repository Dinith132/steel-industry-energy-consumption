"""Compare SHAP vs LIME attribute rankings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import spearmanr

from .paths import xai_dir


def compare_shap_lime(cfg: dict[str, Any], model_type: str, window_size: int) -> dict[str, float]:
    base = xai_dir(cfg, "attributes")
    shap_path = base / f"shap_{model_type}_win{window_size}.csv"
    lime_path = base / f"lime_{model_type}_win{window_size}.csv"
    if not shap_path.exists() or not lime_path.exists():
        raise FileNotFoundError("Run xai_attributes first")

    shap_df = pd.read_csv(shap_path).set_index("attribute")
    lime_df = pd.read_csv(lime_path).set_index("attribute")
    merged = shap_df.join(lime_df, how="inner")
    if len(merged) < 2:
        return {"spearman": float("nan"), "n_features": len(merged)}

    corr, _ = spearmanr(merged["mean_abs_shap"], merged["mean_abs_lime"])
    return {"spearman": float(corr), "n_features": len(merged)}


def compare_all(cfg: dict[str, Any], windows: list[int] | None = None) -> pd.DataFrame:
    windows = windows or cfg.get("xai", {}).get("sample_windows", [8, 24, 168])
    rows = []
    for model_type in cfg["models"]:
        for w in windows:
            try:
                r = compare_shap_lime(cfg, model_type, w)
                rows.append({"model_type": model_type, "window_size": w, **r})
            except FileNotFoundError:
                pass
    df = pd.DataFrame(rows)
    out = xai_dir(cfg) / "shap_lime_correlation.csv"
    df.to_csv(out, index=False)
    return df
