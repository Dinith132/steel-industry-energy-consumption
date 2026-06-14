# Final Research Findings
## Understanding LSTM Behaviors using XAI — Steel Industry Energy Consumption

**Dataset:** UCI DAEWOO steel plant (2018)  
**Tracks:** Hourly (36 models) + 15-minute (27 models) = **63 LSTM configurations**  
**XAI methods:** SHAP · Integrated Gradients · Memory erasure · Fidelity perturbation  
**Scope:** Full audit of **965 files** under `Final-results/Final/` — every behavior CSV, metric, and plot reviewed

---

## Executive summary (read this first)

This research asks: *When an LSTM predicts steel-plant energy consumption, what does it actually use from its input window?*

We did not stop at RMSE. We ran four complementary XAI tests on all 63 trained models. Three conclusions are robust across the full experiment:

### Finding 1 — Recency bias (Single & Double LSTM)
Unidirectional LSTMs concentrate Integrated Gradients on the **most recent 1–2 timesteps**. At hourly window 24, the last step receives **14–37× more attribution** than the oldest step. **79%** of single/double models have >50% of IG mass in the last 25% of the window. When the window grows to 168–672 hours, recency_ratio approaches **1.0** — the model *ignores* weeks of history in attribution, even though erasure tests show those steps still matter for accuracy.

### Finding 2 — Boundary sensitivity (BiLSTM) — your key visual result
Bidirectional LSTMs behave **fundamentally differently**. **67%** (14/21) of bidir models are classified **U-shaped** vs only **10%** (2/21) for single LSTM. At win24 hourly, bidir step 0–1 attribution is **2× higher** than single (0.00109 vs 0.00063 at step 0). The middle of the window (steps 3–14) is a flat "dead zone." At win168, bidir first-5-step mean IG = **0.00225** vs middle mean = **0.000004** — a **560×** difference before the final spike.

This is the architecture-level discovery: BiLSTM reads forward *and* backward, so it anchors on **window boundaries**, not the full sequence.

### Finding 3 — Autoregressive dominance (SHAP + Fidelity)
**Usage_kWh** (past energy) is the #1 SHAP feature in **57%** of all models. For the best hourly model (double win24), SHAP ranks: Usage_kWh (0.087) > Lagging_Current_Power_Factor (0.060) > Leading_Current_Power_Factor (0.055). Zeroing top features raises RMSE by mean **11.8 kWh**. The LSTM is primarily an **autoregressive forecaster**, not a full multivariate physical model of the plant.

**Best accuracy:** Hourly double win24 → RMSE **8.90 kWh**, R² **0.91**. 15min single win96 → RMSE **8.19 kWh**, R² **0.93**. Both use ~24 hours of effective context — but XAI shows they do not *evenly* use all of it.

---

## Method (what we measured)

| Method | Question it answers | Output per model |
|--------|---------------------|------------------|
| **SHAP** | Which *features* matter? | Top feature rankings |
| **Integrated Gradients** | Which *timesteps* matter? | Attribution curve over window |
| **Memory erasure** | Does removing *old* history hurt accuracy? | RMSE vs % oldest erased |
| **Fidelity** | Does zeroing top SHAP features break predictions? | RMSE delta per feature |

**IG convention:** step 0 = oldest timestep in window; highest step = most recent (prediction time).

**Derived metrics (from IG curves):**
- `recency_ratio` — fraction of IG mass in last 25% of window (>0.5 = recency-dominant)
- `start_ratio` — fraction in oldest 25% (high in bidir = U-shape start)
- `u_shape_score` — (start + end mass) / middle mass
- `ig_pattern` — classified: recency_dominant / u_shaped / flat / mixed

---

## 1. Prediction performance

### Best models
| Track | Architecture | Window | RMSE (kWh) | R² | Notes |
|-------|-------------|--------|------------|-----|-------|
| Hourly | **Double LSTM** | 24 h | **8.90** | 0.91 | Best overall hourly |
| 15min | **Single LSTM** | 96 steps (24 h) | **8.19** | 0.93 | Best 15min |

### Performance patterns
- **Worst → best improvement:** Hourly RMSE drops from 14.41 (single win1) to 8.90 — **38% reduction** by tuning window + stack.
- **Sweet spot:** ~24 hours of context (win24 hourly, win96 15min). Shorter windows lack context; longer windows (336–672) do **not** improve RMSE and increase recency bias.
- **Architecture:** Double LSTM wins hourly; Single wins 15min. BiLSTM is competitive but rarely best — its value here is **interpretability** (reveals boundary memory), not peak accuracy.

**Figures:** `plots/rmse_heatmap_hourly.png`, `plots/rmse_heatmap_15min.png`, `plots/rmse_vs_recency.png`

---

## 2. SHAP — feature-level behavior

### Top drivers (stable across 56/63 sane SHAP runs; 7 long-window runs excluded due to Kernel SHAP noise)

| Rank | Feature | Interpretation |
|------|---------|----------------|
| 1 | **Usage_kWh** | Autoregressive — past consumption predicts future |
| 2 | Load_Type, WeekStatus, Day_of_week | Shift/calendar context |
| 3 | NSM, Lagging/Leading current & power factor | Electrical load sensors |
| 4 | CO2(tCO2) | Secondary environmental signal |

**Example — best hourly model (double win24):**
```
Usage_kWh                      0.087
Lagging_Current_Power_Factor   0.060
Leading_Current_Power_Factor   0.055
Lagging_Current_Reactive...    0.054
NSM                            0.049
```

**Fidelity check:** Zeroing Usage_kWh on double win24 → +**10.76 kWh** RMSE. On 15min single win1 → +**23.7 kWh** (strongest autoregressive dependence).

**Research implication:** For energy forecasting, the LSTM's "reasoning" is largely *"recent kWh → next kWh"* with electrical features as modifiers. This is honest and defensible — but it means the model is **not** exploiting the full sensor suite equally.

**Figures:** `plots/shap_usage_kwh_top1_pct.png`, `plots/shap_top_features_sane.png`, `plots/fidelity_heatmap_hourly.png`, `plots/fidelity_heatmap_15min.png`

---

## 3. Integrated Gradients — temporal memory (core contribution)

### 3.1 Hourly win24 — direct evidence from IG CSVs

| Step region | Single LSTM | Double LSTM | BiLSTM |
|-------------|-------------|-------------|--------|
| Step 0 (oldest) | 0.000628 | 0.000463 | **0.001092** |
| Steps 8–15 (middle) | 0.001127 | 0.000559 | **0.000492** |
| Step 23 (newest) | **0.017562** | **0.017046** | **0.015684** |
| End / Start ratio | **28×** | **37×** | **14×** |
| recency_ratio | 0.651 | 0.754 | 0.699 |
| start_ratio | 0.063 | 0.062 | **0.130** |

**How to read this:** Single/double curves are flat for steps 0–20, then explode at step 22–23. BiLSTM adds a visible bump at steps 0–2 (backward pass reads the sequence start), then flat middle, then end spike.

**Visual confirmation:** `plots/ig_compare_win24_hourly.png` — green (bidir) line lifts at step 0–1; blue/orange stay near zero until step ~15.

### 3.2 Window size effect — "Long window ≠ long memory"

| Window (hourly) | Single recency_ratio | Pattern |
|-----------------|---------------------|---------|
| 4 | 0.369 | flat/mixed |
| 8 | 0.496 | mixed |
| 24 | 0.651 | recency_dominant |
| 48 | 0.814 | mixed |
| 168 | 0.997 | mixed (effectively all on last step) |
| 672 | 1.000 | mixed |

Mean recency_ratio: win ≤ 8 → **0.332**; win ≥ 168 → **0.925**.

**Interpretation:** Giving the LSTM more history does not spread attention — it **concentrates** it on the final timestep. The extra capacity is wasted for temporal attribution (though erasure shows distant steps still affect weights indirectly).

**Figures:** `plots/recency_vs_window.png`, `plots/ig_facet_key_windows_hourly.png`, `plots/recency_heatmap_hourly.png`

### 3.3 BiLSTM U-shape at long windows (win168)

Bidir win168 hourly IG:
- First 5 steps mean: **0.002247**
- Middle steps (50–118) mean: **0.000004**
- Last 5 steps mean: **0.005613**

The U-shape **persists** even when the window spans a full week. BiLSTM consistently uses window **boundaries** as anchors.

### 3.4 IG pattern classification (all 63 models)

| Pattern | Single | Double | Bidir |
|---------|--------|--------|-------|
| **u_shaped** | 2 | 5 | **14** |
| recency_dominant | 6 | 2 | 4 |
| mixed | 12 | 12 | 2 |
| flat | 1 | 2 | 1 |

**Figures:** `plots/ig_pattern_counts.png`, `plots/u_shape_by_stack.png`, `plots/recency_ratio_by_stack.png`

---

## 4. Memory erasure — causal validation

Procedure: Replace oldest 0%, 25%, 50%, 75% of window with training means → measure RMSE.

### Hourly win24
| Architecture | Baseline RMSE | After 25% erase | Δ RMSE | After 75% erase |
|-------------|---------------|-----------------|--------|-----------------|
| Single | 9.57 | 11.41 | **+1.84** (+19%) | 13.91 (+45%) |
| Double | 8.90 | 10.94 | **+2.04** (+23%) | — |
| Bidir | 9.54 | 11.61 | **+2.07** (+22%) | — |

### Population-level (averaged across all windows)
- **Hourly:** Bidir most sensitive to erasing old data (avg Δ up to ~4.0 kWh at 75% erase) — consistent with U-shape (start of window matters).
- **15min:** Double most sensitive; bidir least at 75% erase.

### IG vs erasure — why they differ
Correlation(recency_ratio, erasure Δ@25%) = **-0.11** (weak).

This is **not a contradiction**:
- **IG** measures *attribution* to each timestep (where the model "looks")
- **Erasure** measures *causal impact* of removing old data (what the model has encoded in weights)

A recency-biased model can still degrade when old data is removed — the LSTM cell state may encode compressed history even when IG does not highlight it.

**Figures:** `plots/erasure_mean_curves.png`, `plots/erasure_delta25_heatmap_hourly.png`, `plots/erasure_delta25_heatmap_15min.png`, `plots/ig_erasure_scatter.png`

---

## 5. Architecture comparison (for thesis/presentation)

| | Single LSTM | Double LSTM | BiLSTM |
|---|-------------|-------------|--------|
| **IG shape (win24)** | End spike only | End spike | **U-shape** (start + end) |
| **Mean recency_ratio** | 0.643 | 0.631 | 0.580 (lowest) |
| **Mean start_ratio** | 0.156 | 0.161 | **0.241** (highest) |
| **U-shaped models** | 10% | 24% | **67%** |
| **Best RMSE (hourly)** | 9.05 (win74) | **8.90 (win24)** | 9.54 (win24) |
| **Role in study** | Baseline | Best predictor | Best for behavior insight |

---

## 6. Hourly vs 15-minute

| Metric | Hourly | 15min |
|--------|--------|-------|
| Mean recency_ratio | 0.662 | 0.559 |
| Best RMSE | 8.90 kWh | 8.19 kWh |
| Best window (effective) | 24 h | 96×15min = 24 h |

15min models show **slightly less** recency concentration (more granular recent steps spread attribution). Both tracks converge on **~1 day** as the useful forecasting horizon.

**Figures:** `plots/recency_heatmap_15min.png`, `plots/ig_compare_win96_15min.png`

---

## 7. Hidden-state geometry — does the U-shape appear internally?

We extracted hidden vectors from **2000 training windows** (hourly win24) and projected them with PCA, colored by operational labels. NumPy forward pass validated against saved RMSE (exact match to 9.571 / 9.542 kWh).

### 7.1 Colored by Load_Type, WeekStatus, Usage_kWh

| Observation | Single LSTM | BiLSTM |
|-------------|-------------|--------|
| **Load_Type** | Heavy overlap between types | **3 horizontal bands** — types 0/1/2 separated left→right along PC1 |
| **WeekStatus** | Partial separation | **C-shaped arc** — weekday vs weekend form distinct manifolds |
| **Usage_kWh** | Vertical gradient (low→high bottom→top) | Horizontal gradient along PC1 |

BiLSTM hidden space organizes by **operating mode** more distinctly than single LSTM — consistent with boundary-sensitive encoding.

### 7.2 Start vs End usage — U-shape internal test

We colored PCA points by Usage_kWh at the **oldest** timestep (start) vs **newest** timestep (end) in the 24h window:

| Panel | Single LSTM | BiLSTM |
|-------|-------------|--------|
| Color by **START** usage | Mixed colors — weak structure | **Clear left→right gradient** (low→high kWh) |
| Color by **END** usage | Clear left→right gradient | Clear left→right gradient |

**Conclusion:** Single LSTM internal state encodes **end-of-window** energy (recency) but not start. BiLSTM encodes **both boundaries** — the IG U-shape is reflected in hidden-state geometry, not only in attribution curves.

### 7.3 What this adds to the research story

| XAI method | What it shows |
|------------|---------------|
| IG | Where the model *attributes* importance (timesteps) |
| Hidden-state PCA | What the model *encodes* internally (compressed memory) |

Together: BiLSTM doesn't just assign IG mass to window start — its hidden vectors **cluster by start-of-window energy levels**, proving backward-pass information is structurally present in internal memory.

**Figures:**
- `plots/hidden_pca_single_vs_bidir_win24.png` — main comparison (6-panel)
- `plots/hidden_pca_start_vs_end_usage_win24.png` — U-shape internal test
- `plots/hidden_pca_single_win24_colored.png`, `plots/hidden_pca_bidir_win24_colored.png`
- Full write-up: `HIDDEN_STATES_ANALYSIS.md`, metrics: `hidden_states_geometry_win24.csv`

**Re-run:** `cd Final && python3 hidden_states_colored.py`

---

## 8. Limitations (state these in viva/presentation)

1. **Kernel SHAP instability** on 7 long-window models (absurd magnitudes >1000) — use IG/fidelity for those.
2. **Usage_kWh in inputs** — autoregressive dominance is partly structural (predicting kWh from past kWh).
3. **Erasure removes oldest only** — does not test removing recent steps (would hurt recency models more).
4. **Fidelity uses zero in scaled space**, not mean imputation.
5. **Single plant, single year** — generalization to other steel plants unknown.
6. **LIME/Spearman** available only on analysis/ subset (~18 models), not all 63.

---

## 9. Presentation script (30-second version)

> "We trained 63 LSTM models on steel plant energy data and asked not just *how accurate* they are, but *how they think*. Using SHAP, Integrated Gradients, and memory erasure, we found three things. First, single and double LSTMs ignore most of their input window and focus on the last hour — recency bias. Second, bidirectional LSTMs are different: they show a U-shape, caring about both the start and end of the window — this is visible in the IG curves and confirmed across 14 of 21 bidir models. Third, past energy usage dominates predictions — the model is largely autoregressive. Our best model achieves 8.9 kWh RMSE hourly, but XAI shows it does not use the full 24-hour window evenly. This matters because black-box accuracy alone hides *what* the model learned."

---

## 10. Figure guide — which plot for which slide

| Slide topic | Use this figure |
|-------------|-----------------|
| **Main results (4-in-1)** | `plots/presentation_deep_dive.png` |
| Architecture IG comparison | `plots/ig_compare_win24_hourly.png` |
| U-shape across window sizes | `plots/ig_facet_key_windows_hourly.png` |
| Recency bias population | `plots/recency_ratio_by_stack.png` |
| Pattern classification | `plots/ig_pattern_counts.png` |
| Long window failure | `plots/recency_vs_window.png` |
| SHAP autoregression | `plots/shap_usage_kwh_top1_pct.png` |
| Erasure validation | `plots/erasure_mean_curves.png` |
| Model selection | `plots/rmse_heatmap_hourly.png` |
| Hidden-state comparison | `plots/hidden_pca_single_vs_bidir_win24.png` |
| U-shape internal test | `plots/hidden_pca_start_vs_end_usage_win24.png` |

---

## 11. Data files

| File | Description |
|------|-------------|
| `master_metrics.csv` | 63 rows — all derived metrics per model |
| `file_manifest.csv` | 965 files catalogued |
| `gaps_report.csv` | 0 missing behavior outputs |
| `FINDINGS.md` | Auto-generated stats summary |
| **`FINAL_FINDINGS.md`** | **This document — expert analysis** |
| `HIDDEN_STATES_ANALYSIS.md` | Hidden-state PCA deep dive |
| `hidden_states_geometry_win24.csv` | Silhouette + PC1 correlation metrics |

---

## Appendix: Analysis workflow performed

1. Inventoried all 965 files in `Final-results/Final/`
2. Parsed 63 × (SHAP + IG + erasure + fidelity) CSVs into `master_metrics.csv`
3. Computed IG pattern metrics (recency_ratio, start_ratio, u_shape_score, classification)
4. Reviewed existing aggregate plots in `findings/plots/` (19 figures)
5. Generated additional plots: `presentation_deep_dive.png`, `shap_top_features_sane.png`
6. **Hidden-state PCA:** colored by Load_Type/WeekStatus/Usage_kWh; single vs bidir comparison; start vs end usage test (`hidden_states_colored.py`)
7. Validated key findings against raw IG CSVs (win24, win168) and erasure curves
8. Cross-checked with `analysis/best_models.json` and per-model behavior PNGs

*Generated by full pipeline analysis — not template text. All numbers traceable to `master_metrics.csv` and behavior CSVs.*
