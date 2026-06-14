# Research Findings — LSTM Behavior Audit (Quick Stats)

> **For the full expert analysis, presentation script, and evidence tables, read [`FINAL_FINDINGS.md`](FINAL_FINDINGS.md).**

Generated from **965** files across hourly and 15min tracks (63 trained models with full behavior outputs).

## Executive summary

We audited 63 LSTM models (single, double, bidirectional) using SHAP, Integrated Gradients,
memory erasure, and fidelity tests on UCI steel plant energy data.

## Inventory

- Total files catalogued: **965**
- Missing behavior CSVs: **0**
- Models in master table: **63**

## Numbered findings

### 1. Recency bias in single and double LSTM
- 79% of single/double models have recency_ratio > 0.5 (IG mass in last 25% of window).
- Mean recency_ratio single/double: **0.637** vs bidir: **0.580**.
- IG curves typically flat on old timesteps with spike at the most recent step.

### 2. Bidirectional boundary sensitivity (U-shape)
- Mean u_shape_score: bidir **16.78** vs single **16.65**.
- Mean start_ratio (oldest 25%): bidir **0.241** vs single **0.156**.
- 67% of bidir models classified as u_shaped vs 10% single.
- BiLSTM attributes importance to window start AND end; middle timesteps contribute less.

### 3. Long window does not mean long memory
- Mean recency_ratio window>=168: **0.925** vs window<=8: **0.332**.
- Larger input windows do not spread IG attribution across more history.

### 4. Autoregressive feature dominance
- **57%** of all models rank Usage_kWh as #1 SHAP feature.
- Mean max fidelity delta when zeroing top features: **11.83** kWh RMSE increase.

### 5. Best prediction accuracy
- Hourly best: **double** window **24** RMSE **8.90** kWh.
- 15min best: **single** window **96** RMSE **8.19** kWh.
- Lower RMSE does not require lower recency_ratio (accuracy and memory depth differ).

### 6. IG–erasure consistency
- Correlation(recency_ratio, erasure delta@25%): **-0.110**.
- Erasure removes *oldest* steps; high recency models may still show moderate delta when old data is erased.
- Use IG + erasure together: IG shows *where* importance lies; erasure shows *causal* impact of removing history.

### 7. Hourly vs 15-minute resolution
- Hourly mean recency_ratio: **0.662**.
- 15min mean recency_ratio: **0.559**.

## Limitations

- Kernel SHAP is noisy on very long windows (672).
- LIME/Spearman available only on analysis/ subset, not all 63 models.
- Fidelity uses zero in scaled space, not mean imputation.
- IG step 0 = oldest timestep; right side = most recent.

## Figure index

- `plots/erasure_delta25_heatmap_15min.png`
- `plots/erasure_delta25_heatmap_hourly.png`
- `plots/erasure_mean_curves.png`
- `plots/fidelity_heatmap_15min.png`
- `plots/fidelity_heatmap_hourly.png`
- `plots/ig_compare_win24_hourly.png`
- `plots/ig_compare_win96_15min.png`
- `plots/ig_erasure_scatter.png`
- `plots/ig_facet_key_windows_hourly.png`
- `plots/ig_pattern_counts.png`
- `plots/recency_heatmap_15min.png`
- `plots/recency_heatmap_hourly.png`
- `plots/recency_ratio_by_stack.png`
- `plots/recency_vs_window.png`
- `plots/rmse_heatmap_15min.png`
- `plots/rmse_heatmap_hourly.png`
- `plots/rmse_vs_recency.png`
- `plots/shap_usage_kwh_top1_pct.png`
- `plots/u_shape_by_stack.png`

## Data files
- `master_metrics.csv` — one row per model, all derived metrics
- `file_manifest.csv` — full file inventory
- `gaps_report.csv` — missing expected behavior outputs