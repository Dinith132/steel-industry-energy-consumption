"""SHAP and LIME attribute importance for LSTM inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from .evaluate import pick_best_model
from .paths import xai_dir
from .train import load_model_for_window
from .windows import get_window_datasets


def _flatten_windows(X: np.ndarray) -> np.ndarray:
    n, w, f = X.shape
    return X.reshape(n, w * f)


def _feature_labels(names: list[str], window: int) -> list[str]:
    return [f"t-{window - 1 - t}_{name}" for t in range(window) for name in names]


def run_shap(
    cfg: dict[str, Any],
    model_type: str,
    window_size: int,
    max_background: int = 100,
    max_explain: int = 50,
) -> pd.DataFrame:
    model = load_model_for_window(cfg, model_type, window_size)
    X_train, X_test, _, _, meta = get_window_datasets(cfg, window_size)
    names = meta["feature_names"]

    bg = _flatten_windows(X_train[:max_background])
    test = _flatten_windows(X_test[:max_explain])

    explainer = shap.KernelExplainer(
        lambda x: model.predict(x.reshape(-1, window_size, len(names)), verbose=0).ravel(),
        bg,
    )
    shap_values = explainer.shap_values(test, nsamples=100)

    mean_abs = np.abs(shap_values).mean(axis=0)
    labels = _feature_labels(names, window_size)
    flat = pd.DataFrame({"feature": labels, "mean_abs_shap": mean_abs})
    flat["attribute"] = flat["feature"].str.replace(r"^t-\d+_", "", regex=True)
    df = flat.groupby("attribute", as_index=False)["mean_abs_shap"].sum()
    df = df.sort_values("mean_abs_shap", ascending=False)

    out = xai_dir(cfg, "attributes") / f"shap_{model_type}_win{window_size}.csv"
    df.to_csv(out, index=False)
    np.save(xai_dir(cfg, "attributes") / f"shap_values_{model_type}_win{window_size}.npy", shap_values)
    return df


def run_lime(
    cfg: dict[str, Any],
    model_type: str,
    window_size: int,
    num_samples: int = 500,
    num_explain: int = 20,
) -> pd.DataFrame:
    from lime.lime_tabular import LimeTabularExplainer

    model = load_model_for_window(cfg, model_type, window_size)
    X_train, X_test, _, _, meta = get_window_datasets(cfg, window_size)
    names = meta["feature_names"]

    flat_train = _flatten_windows(X_train)
    flat_test = _flatten_windows(X_test[:num_explain])

    explainer = LimeTabularExplainer(
        flat_train,
        feature_names=_feature_labels(names, window_size),
        mode="regression",
        verbose=False,
    )

    importances = np.zeros(len(names))
    counts = 0
    for i in range(len(flat_test)):
        exp = explainer.explain_instance(
            flat_test[i],
            lambda x: model.predict(
                x.reshape(-1, window_size, len(names)), verbose=0
            ).ravel(),
            num_features=len(names) * window_size,
            num_samples=num_samples,
        )
        for feat, weight in exp.as_list():
            for j, name in enumerate(names):
                if name in feat:
                    importances[j] += abs(weight)
        counts += 1

    importances /= max(counts, 1)
    df = pd.DataFrame({"attribute": names, "mean_abs_lime": importances})
    df = df.sort_values("mean_abs_lime", ascending=False)

    out = xai_dir(cfg, "attributes") / f"lime_{model_type}_win{window_size}.csv"
    df.to_csv(out, index=False)
    return df


def run_attributes_for_windows(cfg: dict[str, Any], windows: list[int] | None = None) -> None:
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
        if best_path.exists():
            windows = list(set(windows + [int(best["window_size"])]))

    xai_cfg = cfg.get("xai", {})
    for w in windows:
        print(f"XAI attributes win{w}")
        if "shap" in xai_cfg.get("attributes", ["shap"]):
            run_shap(cfg, model_type, w, max_background=xai_cfg.get("shap_background_samples", 100))
        if "lime" in xai_cfg.get("attributes", ["lime"]):
            run_lime(cfg, model_type, w, num_samples=xai_cfg.get("lime_num_samples", 500))
