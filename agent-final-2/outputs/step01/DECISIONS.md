# Step 01 — Results

**Quality gate:** PASS

- Rows: 19735, 2016-01-11 17:00:00 → 2016-05-27 18:00:00
- Interval: 10 min, gaps: 0
- Null handling: No nulls after strip — no imputation needed
- Target skew: 3.3861, zeros: 0.0%
- rv1==rv2 on all rows: True → rv1 and rv2 are identical random UCI placeholders — drop in step02

## Highly skewed (|skew|>2)
['Appliances', 'lights']

## Next
**step02_correlation — redundancy, target corr, drop list**