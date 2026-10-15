# nonuniform-width-yam64h32
## Hypothesis
Distillation arm; the same lever as `nonuniform-width` (wide late layers on an a0.25_d8 trunk, `_l<j>x<alpha2>`,
see that branch's notes for the timing map), on the fastest 64-band front end. On yam64h32 the a0.25_d8 ->
a0.50_d8 step buys 0.667 (two-seed mean, 0.658 / 0.676) -> 0.687 and costs 7.05x -> 6.06x (logged). Random-weight
timing (mmt, 200 s, GPU, this session): a0.25_d8 7.06x, `a0.25_d8_l5x0.50` 6.76x, `a0.25_d8_l7x1.0` 6.50x,
a0.50_d8 5.61x.

Students (teacher `v4-ft-ps-e60-moderate`, rung C, WSD 7000-56000 + stop rule 1.3, seed 1, buzz+rain+human,
yam64h32, judged on the mmt re-export): `yam64h32:a0.25_d8_l5x0.50`, `yam64h32:a0.25_d8_l7x1.0`.

Prediction: both >= 0.685 at 6.5-6.8x, above everything at that speed or faster (yam64h32:a0.25_d8 0.676 s2 at
7.05x; yam64h32:a0.50_d8 0.687 at 6.06x is slower).
Falsifier: within 0.01 of the a0.25_d8 two-seed mean (0.667), i.e. the width gain is in the early layers.
Matched comparators: `fe_C_yam64h32_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt` (0.658; seed 2 0.676) and
`fe_C_yam64h32_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt` (0.687).

## Changes
`exp/nonuniform-width`'s `05_distill` code and `chain.sh` as a patch; run as `chain.sh yam64h32`, chained from
`../nonuniform-width/chain.sh` in one job.

## Results
