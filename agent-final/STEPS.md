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

## Step 06 — XAI (pending)

After retrained models on clean features.
