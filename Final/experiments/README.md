# Optional experiments (not required for main thesis pipeline)

Use separate configs here so `Final/hourly` and `Final/15min` outputs are never overwritten.

Examples:
- `config_remove_co2.yaml` — drop CO2 column before training
- `config_log1p.yaml` — apply log1p before MinMax scaling

Copy a base config from `../configs/` and set `experiment_name: Final/experiments/<name>`.
