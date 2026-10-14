# yam64h32-s2
## Hypothesis
Distillation arm, the repeat the rule owes (`ladder_record.py repeats`). `yam64h32` (2026-10-14, seed 1) put
`yam64h32:a0.25_d8` at 0.658 / 6.81x (WSD 56000, mmt), more than 0.02 over everything at its speed or faster
(`fast32h16:a0.25_d8_f2` 0.630 / 6.83x, `fast32h16:a0.25` 0.612-0.616 / 7.3x). It doesn't count until a
second seed agrees.

This run repeats it at seed 2: same teacher, rung C, front end, arch, classes (buzz + rain + human), budgets
(WSD 7000 / 14000 / 28000 / 56000, judged at 56000, `mmt` export). The matched control is seed 1.

Prediction: seed 2 within seed noise (0.01-0.02) of 0.658, so still above 0.650 (the 0.02 bar over 0.630).
Falsifier: seed 2 at or under 0.650.

## Changes
None to the method. `05_distill/` carries `exp/yam64h32`'s code as a patch (`mmt` front end, `reexport.py`,
`yam64h32` registered). `chain.sh`: `main.py --rung C --seed 2 --wsd 7000,14000,28000,56000` on the student,
then `reexport.py --fe-impl mmt --suffix _mmt` on its seed-2 branches.

## Results
Step-budget curve, seed 2 (V pool vs teacher; headline = sensitivity_exclquiet @ FPR 0.005, mmt export):

| budget | hit@K | dK | headline (mmt) | x YAM 200s |
|---|---|---|---|---|
| 7000 | 58.23 | | 0.635 | 6.98 |
| 14000 | 61.07 | +2.83 | 0.649 | 7.04 |
| 28000 | 62.23 | +1.16 | 0.665 | 7.08 |
| 56000 | 62.89 | +0.66 | 0.676 | 7.05 |

Seed 1 at the same budgets: 0.637 / 0.643 / 0.655 / 0.658. The stop rule (hit@K gain < 1.3 twice) fires at
56000, the seed-1 budget, so judged at 56000 as planned.

`results.py` seed 1 vs seed 2 at wsd56000_mmt: 0.658 -> 0.676 (+0.018 +- 0.006 eval-sampling SD); inclusive
0.542 -> 0.562. Per fold: 1_29 +0.043 +- 0.016, 53 +0.001 +- 0.011, 1_11 +0.045 +- 0.016, 1_143 +0.005 +- 0.014,
1_37 -0.002 +- 0.013. Tiers: loud +0.002, untagged +0.015, background +0.028, quiet +0.011.
No fold missed FPR 0.005.

Speed 7.05x YAMNet (200 s), against seed 1's 6.81x (same graph; the spread is timing noise).

## Conclusion
The repeat holds: seed 2 at 0.676 is above both the 0.650 falsifier and seed 1. Two-seed mean
0.667 at ~6.9x. That beats `fast32h16:a0.25_d8_f2` (0.630 at 6.83x) by +0.037 and ties
`twofast32:a0.25_d8` (0.668 at 5.85-6.14x) while running ~1.15x faster. So `yam64h32:a0.25_d8` takes the
~7x point of the frontier on two seeds.
The two seeds differ by 0.018. That is about the size of the gaps the frontier is being read on, so single-seed
comparisons at this rung need a margin of at least 0.02. The seed-to-seed difference sits on 1_29 and 1_11; the
other three folds agree within 0.005.
