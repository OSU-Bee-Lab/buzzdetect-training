# fast32h16-a050-s2
## Hypothesis
Seed-2 repeat of `fast32h16:a0.50` (rung B, buzz+rain+human), flagged by `ladder_record.py repeats` (+0.588 at rung B, +0.02 over every student at its speed or faster, no second seed).
Prediction: seed 2 lands within 0.03 of seed 1 (0.588) at the 56000 WSD budget.
Falsifier: seed 2 differs from seed 1 by > 0.03, so the single-seed gain is not trustworthy.
## Changes
None; seed 2 only.

## Results
Seed 2 at WSD 56000: headline 0.571 (seed 1 0.588; diff -0.017 ± 0.011, eval sampling only). Curve: 7000 0.605, 14000 0.612, 28000 0.603, 56000 0.571 (headline falls with budget; hit@K flat after 14000). Two-seed mean at 56000: 0.580.
Per tier (s1 vs s2): loud -0.004, untagged +0.012, background +0.045, quiet +0.030 in favour of s1; no fold missed fpr 0.005.

## Conclusion
Prediction held (|diff| 0.017 < 0.03); falsifier not met. The single-seed rung-B number is repeatable within noise; two-seed mean 0.580 is the estimate for fast32h16:a0.50.
