# Pipeline decision log (data-first)

---

## Step 01 — Distribution audit ✅ EXECUTED

**Script:** `pipeline/step01_distributions.py`  
**Outputs:** `outputs/step01/` (7 plots, `numeric_profile.csv`, `DECISIONS.md`)

| Check | Result |
|-------|--------|
| Quality gate | **PASS** |
| Zero-inflated | Leading reactive **67%**, CO2 **60%** |
| Target | right-skewed (skew=1.20), 0% zeros |

---

## Step 02 — Correlation & leakage ✅ EXECUTED

**Script:** `pipeline/step02_correlation.py`  
**Outputs:** `outputs/step02/`

| Finding | Action |
|---------|--------|
| CO2 ↔ target r=**0.988** | **DROP** — leakage |
| NSM ↔ hour r=**0.999** | **DROP** |
| Leading reactive ↔ Leading PF r=-0.94 | **DROP** Leading reactive |
| Lagging reactive ↔ target r=0.90 | **KEEP** (physical, not CO2-style leakage) |
| VIF all < 2.1 | OK |

---

## Step 03 — Feature selection & preprocess ✅ EXECUTED

**Script:** `pipeline/step03_feature_selection.py`  
**Outputs:** `outputs/step03/{15min,hourly}/`

**Final feature set (9 columns):**
```
Usage_kWh, Lagging_Current_Reactive.Power_kVarh, Lagging_Current_Power_Factor,
Leading_Current_Power_Factor, WeekStatus, Day_of_week, Load_Type, hour_sin, hour_cos
```

**Post-clean validation:** PASS — no input feature with |r|>0.95 vs target

| Track | Rows |
|-------|------|
| 15min | 35,040 |
| hourly | 8,760 |

---

## Step 04 — Windows ✅ EXECUTED

**Script:** `pipeline/step04_windows.py`  
All windows OK on **9 features** (was 10 with junk/leakage).

Recommended: hourly win24, 15min win96.

---

## Step 05 — Training ✅ notebooks ready

**Colab + TensorFlow.** No SHAP in these notebooks.

1. `notebooks/preprocess.ipynb` → `outputs/preprocess/`
2. `notebooks/train_hourly.ipynb` → `outputs/train/hourly/` (36 models: 3 stacks × 12 windows)
3. `notebooks/train_15min.ipynb` → `outputs/train/15min/` (27 models: 3 stacks × 9 windows)

Set `USE_COLAB = True`, `FORCE_RETRAIN = False` to skip already saved `.keras` files.

---

## Step 06 — XAI ✅ notebook ready

**After training completes.** Same methods as old `Final/behavior_analysis.ipynb`, new paths.

**Notebook:** `notebooks/behavior_analysis.ipynb` → `outputs/behaviors/{hourly,15min}/`

| Part | Method | Output |
|------|--------|--------|
| 1 | SHAP | `behaviors/shap/{stack}/winN.csv` + `.png` |
| 2 | LIME + Spearman vs SHAP | `behaviors/lime/...`, `behaviors/shap_lime/...` |
| 3 | Integrated Gradients | `behaviors/ig/...` |
| 4 | Memory erasure | `behaviors/erasure/...` |
| 5 | Fidelity (SHAP top-k) | `behaviors/fidelity/...` |
| 6 | Architecture IG overlay | `compare_ig_win24.png` |
| 7 | Hidden states PCA (best model) | `hidden_states_pca.png` |

Reads models from `outputs/train/{track}/`, data from `outputs/preprocess/{track}/`.  
Set `FORCE_RECOMPUTE = False` to skip existing outputs.

**LIME fix (2026-06):** the original LIME aggregation matched feature *names* in `exp.as_list()` strings and never matched → all-zero LIME → NaN Spearman. Fixed to use `exp.as_map()` indices with `feature = idx % n_features`. Re-run Part 2 with `FORCE_RECOMPUTE_LIME = True` (recomputes only `lime/` + `shap_lime/`). Part 7 now also writes `hidden_states_coords.csv` (PCA coords + usage/hour/load labels) and a usage-colored PCA.

---

## Step 07 — Presentation findings ✅ notebook ready

**Local / CPU only** (pandas + matplotlib, no TensorFlow). Reads the result CSVs and renders slide-ready figures + written conclusions.

**Notebook:** `notebooks/presentation_findings.ipynb` → `outputs/findings/plots/`

| Part | Slide | Figure |
|------|-------|--------|
| 1 | Performance + plateau | `p1_rmse_heatmaps.png`, `p1_rmse_vs_window.png` |
| 2 | Recency bias (IG) | `p2_ig_horizon.png`, `p2_recency_ratio.png` |
| 3 | Architecture fingerprint (headline) | `p3_ig_fingerprint.png`, `p3_erasure_bars.png` |
| 4 | Feature importance + faithfulness | `p4_shap_importance.png`, `p4_faithfulness.png` |
| 5 | SHAP vs LIME (Spearman) | `p5_shap_lime_spearman.png` (needs fixed LIME) |
| 6 | Hidden-state map | `p6_hidden_states.png` (needs `hidden_states_coords.csv`) |
| 7 | Limitations + conclusions | — |

Headline results: best single LSTM RMSE ≈ 8.3–8.8 kWh (R² 0.91–0.93); accuracy plateaus by win16–36; IG + erasure agree that unidirectional LSTMs ignore old history at long windows while bidirectional keeps a U-shape; `Usage_kWh` is the #1 feature in 41/63 models (no CO2 leakage); SHAP top feature is the most-damaging in ~76% of models.

Parts 5 and 6 show graceful placeholders until the fixed `behavior_analysis.ipynb` Parts 2 and 7 are re-run on Colab and the new files copied into `Final-results`.
