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
