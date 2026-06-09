"""Synthesize explanation report for thesis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .evaluate import pick_best_model, results_dir
from .paths import get_output_root, xai_dir
from .xai_compare import compare_all


def _df_to_md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def build_report(cfg: dict[str, Any]) -> Path:
    root = get_output_root(cfg)
    metrics_path = results_dir(cfg) / f"{cfg['track']}_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    best = pick_best_model(cfg, metrics)

    lines = [
        f"# LSTM + XAI Explanation Report — {cfg['track']}",
        "",
        "## Best forecasting model (test RMSE in kWh)",
        f"- Model: **{best['model_type']}**",
        f"- Window size: **{best['window_size']}**",
        f"- RMSE (kWh): **{best['rmse_kwh']:.4f}**",
        "",
        "## Top models (by RMSE kWh)",
        "",
        _df_to_md(
            metrics.nsmallest(5, "rmse_kwh")[
                ["model_type", "window_size", "rmse_kwh", "mae_kwh", "r2_kwh", "wia_kwh"]
            ]
        ),
        "",
    ]

    attr_dir = xai_dir(cfg, "attributes")
    shap_file = attr_dir / f"shap_{best['model_type']}_win{best['window_size']}.csv"
    lime_file = attr_dir / f"lime_{best['model_type']}_win{best['window_size']}.csv"
    if shap_file.exists():
        shap_df = pd.read_csv(shap_file).head(10)
        lines.extend(["## Top attributes (SHAP)", "", _df_to_md(shap_df), ""])
    if lime_file.exists():
        lime_df = pd.read_csv(lime_file).head(10)
        lines.extend(["## Top attributes (LIME)", "", _df_to_md(lime_df), ""])

    ig_plot = xai_dir(cfg, "timesteps") / f"ig_horizon_{best['model_type']}_win{best['window_size']}.png"
    if ig_plot.exists():
        lines.extend([
            "## Time-step importance (Integrated Gradients)",
            "",
            f"Memory horizon plot: `{ig_plot.name}`",
            "",
            "Interpretation: higher attribution at a lag means the LSTM relies more on that past time step when predicting Usage_kWh.",
            "",
        ])

    corr_path = xai_dir(cfg) / "shap_lime_correlation.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        lines.extend(["## SHAP vs LIME agreement (Spearman)", "", _df_to_md(corr), ""])
    else:
        try:
            corr = compare_all(cfg)
            lines.extend(["## SHAP vs LIME agreement (Spearman)", "", _df_to_md(corr), ""])
        except Exception:
            pass

    lines.extend([
        "## Discussion prompts",
        "",
        "- Compare hourly vs 15-min tracks: which resolution gives lower RMSE?",
        "- Does BiLSTM improve over single-layer for the same window?",
        "- Do SHAP and LIME agree on reactive power / NSM / load type?",
        "- Does the IG memory-horizon curve decay quickly (short memory) or stay flat (long memory)?",
        "",
        "_Phase 2: run fidelity perturbation and hidden-state clustering via `07_behavior_audit`._",
    ])

    out = root / "explanation_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    return out
