# agent-final-2 — Appliance energy LSTM pipeline

UCI **Appliances energy prediction** dataset (`energydata_complete.csv`).

**Target:** `Appliances` (Wh per 10-minute interval)

## Run order

```bash
cd agent-final-2
python pipeline/step01_distributions.py
python pipeline/step02_correlation.py
python pipeline/step03_preprocess.py
python pipeline/step04_windows.py
```

Then open notebooks in order:

1. `notebooks/preprocess.ipynb` — review EDA + saved data
2. `notebooks/train_10min.ipynb` / `train_hourly.ipynb` — LSTM training (Colab OK)
3. `notebooks/behavior_analysis.ipynb` — SHAP, LIME, IG, erasure, fidelity

Decision log: **`STEPS.md`**

## vs agent-final (steel)

| | Steel | Appliances |
|---|-------|------------|
| Target | Usage_kWh | Appliances |
| Native resolution | 15min | **10min** |
| Leakage drop | CO2 | rv1, rv2 (random UCI cols) |
| Hourly agg | mean | **sum** Appliances/lights |

Raw CSV: `../energydata_complete.csv`
