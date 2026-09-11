# low-delta

## Hypothesis
Reducing EarlyStopping min_delta from 0.01 to 0.002 prevents premature stopping on bad-luck initializations, reducing training variance without sacrificing mean performance.

## Changes
- `03_train/train.py`: min_delta 0.01 → 0.002; removed Dropout(0.2) (same as ls02-repro, to isolate the delta effect)

## Results

**Control** (min_delta=0.01, no dropout, n=8):
`[0.107, 0.160, 0.196, 0.228, 0.231, 0.243, 0.250, 0.270]`
mean=0.211, median=0.230, std=0.054

**Low-delta** (min_delta=0.002, no dropout, n=8):
`[0.190, 0.204, 0.219, 0.221, 0.231, 0.236, 0.249, 0.250]`
mean=0.225, median=0.226, std=0.021

Statistical tests:
- Mann-Whitney (mean rank difference): p=0.48 — not significant
- Permutation test (mean diff +0.014): p=0.26 — not significant
- F-test on variances: **p=0.023** — significant; 6.6× lower variance

## Conclusion
Lower min_delta does not improve expected performance but eliminates bad-convergence runs (control floor: 0.107; low-delta floor: 0.190). The variance reduction is statistically significant (F-test p=0.023). Adopting min_delta=0.002 as default makes single-run and small-n experiments trustworthy enough to act on. Combined with patience=50 and epochs=400, this becomes the new training default.
