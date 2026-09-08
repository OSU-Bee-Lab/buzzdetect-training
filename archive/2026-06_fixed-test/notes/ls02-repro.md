# ls02-repro
## Hypothesis
Replicate label-smooth-02 (no dropout, ls=0.2, lr=0.002) on current medium set to establish a clean baseline.

## Changes
Removed Dropout(0.2) from train.py (main has it baked in from dropout-probe). Otherwise identical to current defaults.

## Results
- Claimed (label-smooth-02, stale set): 0.2993
- This run (exp_ls02_repro_v1, current set): 0.160

## Conclusion
All null-commit results are invalid. Consistent with hyperparam-sweep observation: dropout=0.2 gives ~20% on current set. Training variance is very high; single runs are unreliable. Expected range on current dataset appears to be ~0.17–0.25 @ 95% precision.
