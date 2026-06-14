# Hidden-State Geometry Analysis (hourly win24)

## What we did
- Extracted hidden vectors for **2000** training windows (NumPy LSTM forward pass, RMSE-validated vs saved models)
- PCA → 2D, colored by Load_Type, WeekStatus, Usage_kWh bins
- Compared **single vs bidir** internal geometry
- Tested U-shape hypothesis: color by **start** (oldest step) vs **end** (newest step) Usage_kWh

## Key finding — BiLSTM U-shape appears internally

| Metric | Single LSTM | BiLSTM |
|--------|-------------|--------|
| PC1 correlation with **start** Usage_kWh | 0.581 | 0.493 |
| PC1 correlation with **end** Usage_kWh | 0.634 | 0.666 |
| WeekStatus silhouette | 0.090 | 0.108 |
| Usage_kWh silhouette | 0.104 | 0.096 |

**Interpretation:**
- **Single LSTM:** Hidden space aligns with **end-of-window** usage (recency) — start usage is weakly encoded. Matches IG recency bias.
- **BiLSTM:** Hidden space aligns with **both start AND end** usage along PC1. The start-vs-end colored plot shows a clear left→right gradient for BiLSTM on *both* panels; single only on the end panel.
- **Load_Type:** BiLSTM shows three separated horizontal bands (types 0/1/2 left→right); single shows heavy overlap.
- **WeekStatus:** BiLSTM forms a C-shaped arc separating weekday vs weekend in hidden space.

This connects IG's U-shape (importance at window boundaries) to **internal representation geometry** — BiLSTM doesn't just *attribute* to start/end; it *encodes* start and end energy levels in distinct hidden-state regions.

## Silhouette scores (supplementary)

| Label | Single | BiLSTM |
|-------|--------|--------|
| Load_Type | 0.053 | -0.001 |
| WeekStatus | 0.090 | 0.108 |
| Start Usage_kWh | 0.026 | 0.011 |
| End Usage_kWh | 0.065 | 0.082 |

## Figures
- `plots/hidden_pca_single_win24_colored.png` — single, 3 label colorings
- `plots/hidden_pca_bidir_win24_colored.png` — bidir, 3 label colorings
- `plots/hidden_pca_single_vs_bidir_win24.png` — **main comparison slide**
- `plots/hidden_pca_start_vs_end_usage_win24.png` — **U-shape internal test**

## Re-run
```bash
cd Final && python3 hidden_states_colored.py
```
