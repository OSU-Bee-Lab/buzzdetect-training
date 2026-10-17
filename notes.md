# l5x050-yam64h32-s2
## Hypothesis
Seed-2 repeat of `yam64h32:a0.25_d8_l5x0.50` (`ladder_record.py repeats` flagged it: +0.067 over every student at its speed
or faster, no second seed). Seed 1 scored 0.683 at ~7.2x, +0.025 ± 0.006 over a0.25_d8 s1 but only +0.007 ± 0.005 over
a0.25_d8 s2 (0.676), so the width gain may be seed noise.
Prediction: seed 2 lands >= 0.675 (two-seed mean >= 0.68), i.e. l5x0.50 is level with a0.50_d8 (0.687) at 1.2x its speed.
Falsifier: seed 2 < 0.670, then the l5x0.50 gain over the a0.25_d8 two-seed mean (0.667) is within seed noise.
Comparators: `fe_C_yam64h32_a0.25_d8_s2_c-buzz-rain-human_wsd56000_mmt` (0.676, same seed) and the seed-1 l5x0.50 row.

## Changes
`05_distill/student.py`, `distill_train.py`, `bench_arch.py` from `exp/nonuniform-width` (the `_l<j>x<alpha2>` arch names). Seed 2 only.

## Results
Student `fe_C_yam64h32_a0.25_d8_l5x0.50_s2_c-buzz-rain-human_wsd56000_mmt` vs same-seed control a0.25_d8 s2:

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.803 | 0.761 | -0.042 | 0.016 | 34 |
| 53 | 0.647 | 0.636 | -0.011 | 0.012 | 32 |
| 1_11 | 0.634 | 0.591 | -0.043 | 0.024 | 32 |
| 1_143 | 0.612 | 0.607 | -0.005 | 0.015 | 26 |
| 1_37 | 0.685 | 0.685 | +0.000 | 0.014 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.676 → this 0.656 (-0.020 ± 0.008)
- inclusive (sensitivity), same thresholds: 0.562 → 0.546 (-0.016)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.808 | 0.820 | +0.012 | 603 |
| untagged | 0.677 | 0.658 | -0.019 | 7485 |
| background | 0.686 | 0.655 | -0.031 | 9412 |
| quiet | 0.203 | 0.198 | -0.005 | 2231 |
| faint | 0.000 | 0.042 | +0.042 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Seed 2 headline 0.656 at 7.19x (200 s). Against a0.50_d8 s1 (0.687): -0.031 ± 0.007. WSD curve 0.630/0.645/0.663/0.656 (7k..56k; peak 28k, plateau).
Two-seed l5x0.50 mean 0.670 (0.683, 0.656) vs a0.25_d8 two-seed mean 0.667: gain +0.003, within seed spread (0.027).

## Conclusion
Prediction (seed 2 >= 0.675) failed; falsifier met (0.656 < 0.670). The seed-1 gain of l5x0.50 over a0.25_d8 was seed noise; widening layers late buys nothing measurable on yam64h32. Seed 2 is -0.020 ± 0.008 under its same-seed control. Do not pursue wide-late further.
