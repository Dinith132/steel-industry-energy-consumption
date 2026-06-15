# agent-final — data-first pipeline

**Rule:** No modeling until data is audited. Each step runs, outputs are reviewed, then the next step is chosen.

## What was wrong before

The old `Final/hourly.ipynb` / `15min.ipynb` pipelines:
- Skipped proper distribution analysis
- Fed **CO2** (r=0.99 with target) → **target leakage**
- Kept **NSM** (r=0.999 with hour) → redundant junk
- Kept **Leading reactive power** (67% zeros, redundant with Leading PF)
- No VIF / feature-feature correlation check

## What we do now

| Step | Script / Notebook | Purpose |
|------|-------------------|---------|
| — | **`notebooks/preprocess.ipynb`** | EDA → drop bad features → save data |
| — | **`notebooks/train_hourly.ipynb`** | Train hourly LSTMs (no SHAP) |
| — | **`notebooks/train_15min.ipynb`** | Train 15min LSTMs (no SHAP) |
| — | **`notebooks/behavior_analysis.ipynb`** | XAI: SHAP, LIME, Spearman, IG, erasure, fidelity (after training) |
| 01 | `step01_distributions.py` | Histograms, skew, zeros, temporal coverage |
| 02 | `step02_correlation.py` | Feature-feature r, VIF, leakage flags |
| 03 | `step03_feature_selection.py` | Drop bad features, add hour_sin/cos, preprocess |
| 04 | `step04_windows.py` | Validate window shapes on clean data |
| 05 | `step05_train.py` | Train (Colab) |

## Run

**Recommended — single notebook (Colab or local):**
```bash
# Open and run all cells:
agent-final/notebooks/preprocess.ipynb
```

**Or step scripts:**
```bash
cd agent-final
python pipeline/step01_distributions.py
python pipeline/step02_correlation.py
python pipeline/step03_feature_selection.py
python pipeline/step04_windows.py
```

Preprocess outputs: **`outputs/preprocess/{15min,hourly}/`**  
Training outputs: **`outputs/train/{15min,hourly}/`**  
XAI outputs: **`outputs/behaviors/{15min,hourly}/`**

### Colab order
1. `preprocess.ipynb`
2. `train_hourly.ipynb` and/or `train_15min.ipynb`
3. `behavior_analysis.ipynb` (after models are saved)

Decision log: **`STEPS.md`**

## Clean feature set (9 columns)

```
Usage_kWh                          (target + autoregressive lag in windows)
Lagging_Current_Reactive.Power_kVarh
Lagging_Current_Power_Factor
Leading_Current_Power_Factor
WeekStatus, Day_of_week, Load_Type
hour_sin, hour_cos                   (replaces NSM)
```

**Dropped:** CO2, NSM, Leading_Current_Reactive_Power_kVarh
