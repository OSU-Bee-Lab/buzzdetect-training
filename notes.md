# fast32h16-d8-repeat
## Hypothesis
Distillation arm, repeat rule. `fe_C_fast32h16_a0.50_d8_s1_c-buzz-rain-human_wsd56000` gained +0.034 over every rung-C student at its speed or faster, on one seed. Headline seed noise on a student is ~0.01-0.02, so the gain counts only if seed 2 of the identical run (same teacher, rung, WSD budgets, front end, width, classes) lands within that noise of seed 1 and still above everything at its speed or faster. Falsified if seed 2 falls back onto the existing frontier (gain <= 0.02).

Matched control: the seed-1 row itself, at the same budget.

## Changes
None. `ladder_record.py repeats`' command, verbatim:
`05_distill/main.py --rung C --seed 2 --wsd 7000,14000,28000,56000 --runs "fast32h16:a0.50_d8:classes=ins_buzz+ambient_rain+human"`

## Results
Seed 2 (`fe_C_fast32h16_a0.50_d8_s2_c-buzz-rain-human_wsd56000`) against seed 1 at the same budget, `tools/results.py` on the eval dirs:

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.728 | 0.739 | +0.011 | 0.018 | 34 |
| 53 | 0.647 | 0.647 | +0.000 | 0.011 | 32 |
| 1_11 | 0.608 | 0.566 | -0.042 | 0.013 | 32 |
| 1_143 | 0.607 | 0.612 | +0.005 | 0.015 | 26 |
| 1_37 | 0.642 | 0.644 | +0.002 | 0.016 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.646 → this 0.642 (-0.004 ± 0.007)
- inclusive (sensitivity), same thresholds: 0.536 → 0.534 (-0.002)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.778 | 0.782 | +0.004 | 603 |
| untagged | 0.649 | 0.641 | -0.008 | 7485 |
| background | 0.642 | 0.650 | +0.008 | 9412 |
| quiet | 0.193 | 0.193 | +0.000 | 2231 |
| faint | 0.000 | 0.000 | +0.000 | 35 |

± SD is eval sampling only; seed noise is not in it.

Step-budget curve, headline (seed 1 / seed 2): 7000 0.624 / 0.612, 14000 0.640 / 0.625, 28000 0.648 / 0.633, 56000 0.646 / 0.642. Speed at 200 s: 2.64x / 2.66x YAMNet.

Interpretation. The two seeds agree at the judged budget: -0.004 ± 0.007, well inside the 0.01-0.02 seed noise. Seed 2 ran 0.012-0.015 below seed 1 at the three shorter budgets and closed the gap only at 56000, where seed 1 had flattened; that spread is the seed noise, seen directly. One fold moved outside its eval SD, `1_11` at -0.042 ± 0.013; with seed noise added (per-fold delta SD ~0.02 on the training arm, unmeasured for students) it is weak evidence of a seed-dependent fold, not a collapse (0.566). The other four folds are within their SDs. No tier moved: all within 0.008.

Against the frontier. The rule as written holds for both seeds: the only rung-C student at 2.66x or faster is `fast32h16 a0.25` (0.612 at 2.75x), so the gain is +0.030 (seed 2) and +0.034 (seed 1), two-seed mean 0.644 at 2.65x. But the margin that excludes the next student is thin: `fast32h16 a0.375` scores 0.639 at 2.63x, 0.005 below and 0.02-0.03x slower. Speed timing on one architecture varies more than that across its own budget rows (a0.375: 2.54-2.63x; a0.50_d8: 2.64-2.67x). So a0.50_d8 and a0.375 are the same frontier point within noise on both axes; neither dominates the other.

## Conclusion
Confirmed as a reproducible number, not as a new frontier step. `fast32h16:a0.50_d8` at rung C is 0.644 (two seeds, 0.646 and 0.642) at 2.65x YAMNet, incl. quiet 0.535; the seeds differ by -0.004 ± 0.007 and no tier moved. It passes the repeat rule (+0.030 over `fast32h16 a0.25`, the only student at its speed or faster), but that pass rests on being 0.02-0.03x faster than `fast32h16 a0.375` (0.639 at 2.63x), which is inside timing noise. Read: depth-8 at width 0.50 and plain width 0.375 land on the same point, about 0.64 at 2.65x; the +0.034 that triggered the repeat was a gain over a0.25, not over the neighbourhood. Where to dig: seed 2 of `fast32h16 a0.375` would say whether the two are distinguishable at all, and `1_11` is the fold to watch for seed sensitivity (-0.042 ± 0.013 here).
