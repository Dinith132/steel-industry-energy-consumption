# Final — Understanding LSTM Behaviors using XAI

UCI Steel Industry Energy Consumption dataset (DAEWOO plant, 2018).

## Submission notebooks (use these)

| Notebook | Resolution | Window sizes |
|----------|------------|--------------|
| [`hourly.ipynb`](hourly.ipynb) | 1-hour (resampled) | 1, 4, 8, 12, 16, 24, 36, 48, 74, 168, 336, 672 |
| [`15min.ipynb`](15min.ipynb) | 15-minute (original) | 1, 4, 8, 16, 24, 48, 64, 96, 672 |

Self-contained Colab notebooks. No `src/` imports.

### How to run

1. Open notebook in Google Colab
2. Run all cells (clone repo, mount Drive, pip install are in the notebook)
3. Outputs save to `MyDrive/Shared-Colab-Storage/Final/hourly/` or `.../15min/`

### What each notebook does

1. Clone `https://github.com/Dinith132/steel-industry-energy-consumption.git` and load `Steel_industry_data.csv`
2. EDA — plots, correlation heatmap
3. Preprocessing — encode, scale, save `data.csv` to Drive
4. Train Single / Double / BiLSTM for each sliding window
5. If model already on Drive → load it (no retrain)
6. Evaluate — RMSE, MAE, R2, WIA in kWh
7. XAI — SHAP + memory horizon (Integrated Gradients) per model, saved to Drive

Set `FORCE_RETRAIN = True` only when you want to redo everything.

### Smoke test

At the top of a notebook, temporarily use:

```python
window_sizes = [8, 24]
```

## Other folders (optional, for development)

- [`src/`](src/) + [`notebooks/`](notebooks/) — modular pipeline with CLI
- [`fullNotebooks/`](fullNotebooks/) — deprecated

## Data

Original CSV comes from the cloned GitHub repo. Also on [UCI](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption).
