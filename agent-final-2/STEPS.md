# Pipeline decision log — agent-final-2

Appliance energy dataset (`../energydata_complete.csv`). Target: **Appliances** (Wh).

---

## Step 01 — Distribution audit ✅

**Script:** `pipeline/step01_distributions.py`  
**Outputs:** `outputs/step01/`

| Check | Result |
|-------|--------|
| Quality gate | **PASS** |
| Rows | 19,735 @ 10-min, Jan–May 2016 |
| Nulls | **None** (strip whitespace on load) |
| Target skew | 3.39, min=10 Wh |
| rv1 == rv2 | Always true → random columns |

---

## Step 02 — Correlation ✅

**Script:** `pipeline/step02_correlation.py`  
**Outputs:** `outputs/step02/`

| Finding | Action |
|---------|--------|
| rv1 ↔ rv2 r=1.0 | **DROP** both |
| T6 ↔ T_out r=0.97 | **DROP T6** |
| T9 ↔ other T r>0.9 | **DROP T9** |
| Target leakage | **None** (max \|r\| ≈ 0.20 with lights) |

---

## Step 03 — Preprocess ✅

**Script:** `pipeline/step03_preprocess.py`  
**Outputs:** `outputs/preprocess/{10min,hourly}/`, manifest in `outputs/step03/`

**Final features (28 columns):** Appliances + 27 inputs (room sensors, weather, lights, hour/dow sin/cos)

| Track | Rows |
|-------|------|
| 10min | 19,735 |
| hourly | 3,290 (sum Appliances/lights, mean sensors) |

Validation: **PASS** — no \|r\|>0.95 input vs target

---

## Step 04 — Windows ✅

**Script:** `pipeline/step04_windows.py`

| Track | Windows |
|-------|---------|
| 10min | 1, 6, 12, 36, 72, 144, 288 |
| hourly | 1, 4, 8, 12, 24, 48, 72, 168 |

All window sizes validated OK for train/test split.

---

## Step 05 — Training

**Notebooks:** `notebooks/train_10min.ipynb`, `notebooks/train_hourly.ipynb`

- 3 stacks × 7 windows × 2 tracks = **42 models**
- Metrics → `outputs/train/{track}/results_metrics.csv`
- Set `FORCE_RETRAIN = False` to skip saved models (Colab-friendly)

Run on Colab or local GPU; not run automatically by pipeline scripts.

---

## Step 06 — XAI

**Notebook:** `notebooks/behavior_analysis.ipynb`

After training: SHAP, LIME, IG, memory erasure, fidelity → `outputs/behaviors/{10min,hourly}/`

---

## Quick run (scripts only)

```bash
cd agent-final-2
python pipeline/step01_distributions.py
python pipeline/step02_correlation.py
python pipeline/step03_preprocess.py
python pipeline/step04_windows.py
```

Then open notebooks for train + XAI.
