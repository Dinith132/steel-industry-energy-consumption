"""CLI entrypoint for the Final pipeline."""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .data import preprocess
from .evaluate import evaluate_all, save_best_model_json
from .explain import build_report
from .paths import sync_to_drive
from .train import train_all
from .xai_attributes import run_attributes_for_windows
from .xai_compare import compare_all
from .xai_timesteps import run_timesteps_for_windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Final LSTM + XAI pipeline")
    parser.add_argument(
        "command",
        choices=[
            "preprocess",
            "train",
            "evaluate",
            "xai_attributes",
            "xai_timesteps",
            "xai_compare",
            "explain",
            "fidelity",
            "hidden_states",
            "all",
        ],
    )
    parser.add_argument("--track", choices=["hourly", "15min"], default="hourly")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--force", action="store_true", help="Retrain even if model exists")
    parser.add_argument("--sync-drive", action="store_true", help="Copy outputs to Drive")
    args = parser.parse_args(argv)

    cfg = load_config(track=args.track, config_path=args.config)

    if args.command == "preprocess":
        preprocess(cfg)
    elif args.command == "train":
        train_all(cfg, force=args.force)
    elif args.command == "evaluate":
        df = evaluate_all(cfg)
        save_best_model_json(cfg, df)
    elif args.command == "xai_attributes":
        run_attributes_for_windows(cfg)
    elif args.command == "xai_timesteps":
        run_timesteps_for_windows(cfg)
    elif args.command == "xai_compare":
        compare_all(cfg)
    elif args.command == "explain":
        build_report(cfg)
    elif args.command == "fidelity":
        from .fidelity import run_fidelity

        run_fidelity(cfg)
    elif args.command == "hidden_states":
        from .hidden_states import run_hidden_state_pca

        run_hidden_state_pca(cfg)
    elif args.command == "all":
        preprocess(cfg)
        train_all(cfg, force=args.force)
        df = evaluate_all(cfg)
        save_best_model_json(cfg, df)
        run_attributes_for_windows(cfg)
        run_timesteps_for_windows(cfg)
        compare_all(cfg)
        build_report(cfg)

    if args.sync_drive:
        sync_to_drive(cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
