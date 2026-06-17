# Step 02 — Correlation decisions

## Leakage (|r| >= 0.95 with target)
none — good

## High corr pairs (|r| >= 0.85)
| feature_a   | feature_b   |      r |
|:------------|:------------|-------:|
| rv1         | rv2         | 1      |
| T6          | T_out       | 0.9748 |
| T7          | T9          | 0.9448 |
| T5          | T9          | 0.9111 |
| T3          | T9          | 0.9013 |
| RH_3        | RH_4        | 0.899  |
| RH_4        | RH_7        | 0.8943 |
| T1          | T3          | 0.8924 |
| T4          | T9          | 0.8894 |
| T3          | T5          | 0.8882 |
| T1          | T5          | 0.8852 |
| RH_7        | RH_8        | 0.884  |
| T7          | T8          | 0.8821 |
| RH_1        | RH_4        | 0.8804 |
| T4          | T7          | 0.8778 |
| T1          | T4          | 0.877  |
| T4          | T5          | 0.8718 |
| T5          | T7          | 0.8706 |
| T8          | T9          | 0.8693 |
| RH_7        | RH_9        | 0.8587 |
| RH_4        | RH_9        | 0.8566 |
| RH_8        | RH_9        | 0.8558 |
| T3          | T4          | 0.8528 |

## DROP list
```
['T6', 'T9', 'rv1', 'rv2']
```

| Feature | Reason |
|---------|--------|
| T6 | Near-duplicate of T_out (r=0.9748) — keep outdoor T_out |
| T9 | Highly correlated with other room sensors (r up to 0.9013) |
| rv1 | Random UCI placeholder — identical to rv2, no physical meaning |
| rv2 | Random UCI placeholder — drop with rv1 |

## KEEP inputs
['Press_mm_hg', 'RH_1', 'RH_2', 'RH_3', 'RH_4', 'RH_5', 'RH_6', 'RH_7', 'RH_8', 'RH_9', 'RH_out', 'T1', 'T2', 'T3', 'T4', 'T5', 'T7', 'T8', 'T_out', 'Tdewpoint', 'Visibility', 'Windspeed', 'lights']

## Engineered
['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']

## Next
**step03_preprocess — apply drops, engineer time, save 10min + hourly tracks**