"""Load YAML pipeline configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

FINAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FINAL_ROOT.parent
STEEL_DATA_DIR = REPO_ROOT / "steel-industry-energy-consumption"


def load_config(track: str | None = None, config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is not None:
        path = Path(config_path)
    elif track == "hourly":
        path = FINAL_ROOT / "configs" / "config_hourly.yaml"
    elif track in ("15min", "15-min"):
        path = FINAL_ROOT / "configs" / "config_15min.yaml"
    else:
        raise ValueError("Provide track ('hourly' or '15min') or config_path")

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    cfg["_final_root"] = str(FINAL_ROOT)
    return cfg
