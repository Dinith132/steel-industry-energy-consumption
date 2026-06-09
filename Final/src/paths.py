"""Output and Drive path helpers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


def get_output_root(cfg: dict[str, Any], use_drive: bool = False) -> Path:
    exp = cfg["experiment_name"]
    if use_drive and _drive_available(cfg):
        root = Path(cfg["drive_base"]) / exp
    else:
        root = Path(cfg["_final_root"]) / "outputs" / cfg["track"]
    root.mkdir(parents=True, exist_ok=True)
    return root


def _drive_available(cfg: dict[str, Any]) -> bool:
    drive_base = cfg.get("drive_base", "")
    return drive_base.startswith("/content/drive") and Path(drive_base).exists()


def model_dir(cfg: dict[str, Any], model_type: str, use_drive: bool = False) -> Path:
    root = get_output_root(cfg, use_drive=use_drive)
    path = root / model_type / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def history_dir(cfg: dict[str, Any], model_type: str, use_drive: bool = False) -> Path:
    root = get_output_root(cfg, use_drive=use_drive)
    path = root / model_type / "history"
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(cfg: dict[str, Any], use_drive: bool = False) -> Path:
    root = get_output_root(cfg, use_drive=use_drive)
    path = root / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def xai_dir(cfg: dict[str, Any], sub: str = "", use_drive: bool = False) -> Path:
    root = get_output_root(cfg, use_drive=use_drive)
    path = root / "xai" / sub if sub else root / "xai"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plots_dir(cfg: dict[str, Any], sub: str = "", use_drive: bool = False) -> Path:
    root = get_output_root(cfg, use_drive=use_drive)
    path = root / "plots" / sub if sub else root / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sync_to_drive(cfg: dict[str, Any]) -> None:
    """Copy local outputs to Drive when running in Colab."""
    if not _drive_available(cfg):
        return
    local = Path(cfg["_final_root"]) / "outputs" / cfg["track"]
    remote = Path(cfg["drive_base"]) / cfg["experiment_name"]
    if not local.exists():
        return
    remote.mkdir(parents=True, exist_ok=True)
    for item in local.rglob("*"):
        if item.is_file():
            rel = item.relative_to(local)
            dest = remote / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or item.stat().st_mtime > dest.stat().st_mtime:
                shutil.copy2(item, dest)


def resolve_data_csv(cfg: dict[str, Any]) -> Path:
    from .config import STEEL_DATA_DIR

    name = cfg.get("data_file", "steel_industry_data_preprocessed_v1.csv")
    local = STEEL_DATA_DIR / name
    if local.exists():
        return local
    alt = Path(cfg["_final_root"]) / "data" / name
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Dataset not found: {name} under {STEEL_DATA_DIR}")
