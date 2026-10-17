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

Both students ran the full WSD ladder (stop rule did not fire; last budget 56000), judged on the mmt re-export,
teacher `v4-ft-ps-e60-moderate`, rung C, seed 1. Speed is the logged 200 s mmt timing (xYAM).

| student | headline (exclquiet @0.005) | xYAM | vs a0.25_d8 s1 (0.658) | vs a0.25_d8 s2 (0.676) | vs a0.50_d8 (0.687) |
|---|---|---|---|---|---|
| `a0.25_d8_l5x0.50` | 0.683 | ~7.2x | +0.025 ± 0.006 | +0.007 ± 0.005 | -0.004 ± 0.006 |
| `a0.25_d8_l7x1.0` | 0.667 | 6.79x | +0.009 ± 0.006 | -0.009 ± 0.006 | -0.020 ± 0.007 |

Per-fold (l5x0.50 vs s1 control, folds 1_29/53/1_11/1_143/1_37): +0.039, -0.011, +0.049, +0.027, +0.023.
(l7x1.0 vs s1 control): -0.011, -0.001, +0.038, +0.002, +0.018. Per-fold vs the s2 control and vs a0.50_d8 are in the
results.py output; eval-sampling SD only, training noise is larger per fold.
Tiers, l5x0.50 vs s1 control: loud +0.008, untagged +0.024, background +0.017, quiet +0.035, faint +0.070 (35 frames).
l7x1.0 vs s1 control: loud +0.006, untagged +0.008, background -0.005, quiet +0.014, faint +0.028.
Step-budget curve (headline at 7000/14000/28000/56000): l5x0.50 0.650/0.664/0.671/0.683; l7x1.0 0.636/0.651/0.661/0.667.

## Conclusion
`a0.25_d8_l5x0.50` is the result: 0.683 at ~7.2x, +0.025 over the seed-1 a0.25_d8 and +0.007 ± 0.005 over seed 2
(weak against seed 2), level with a0.50_d8 (-0.004 ± 0.006) while ~1.2x faster than it (6.06x) and as fast as the
a0.25_d8 baseline (7.05x). Prediction (>= 0.685 for both) half-met: l5x0.50 0.683 just under, l7x1.0 0.667 failed.
Falsifier met for l7x1.0 (0.667 equals the two-seed mean 0.667): widening the last layers fully (x1.0 from layer 7)
buys nothing here; the narrower, earlier l5x0.50 is the useful one. Same shape as twofast32 (l5x0.50 best). Seed-2
repeat of l5x0.50 is flagged by `ladder_record.py repeats`.
