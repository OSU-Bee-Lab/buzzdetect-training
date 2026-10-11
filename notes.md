# fast32h16-a025-repeat
## Hypothesis
Distillation arm, repeat rule. `fe_C_fast32h16_a0.25_s1_c-buzz-rain-human_wsd56000` (0.612 at 2.75x YAMNet)
is the fastest rung-C student, so `ladder_record.py frontier` lists it as awaiting a repeat: nothing at its
speed or faster exists, and its whole headline is the "gain". It is also the control `fused-blocks` judged
`a0.25_d8_f2` against (0.630 at 2.73x, +0.018 +- 0.008 on one seed each), so the fast end of the frontier
rests on single seeds. Headline seed noise on a student is ~0.01-0.02: the point counts if seed 2 of the
identical run (same teacher, rung, WSD budgets, front end, width, classes) lands within that noise of
seed 1. Falsified if seed 2 differs from seed 1 by more than 0.02 at the judged budget (56000).

Matched control: the seed-1 row itself, at the same budget.

## Changes
None. `ladder_record.py frontier`'s repeat command, verbatim:
`05_distill/main.py --rung C --seed 2 --wsd 7000,14000,28000,56000 --runs "fast32h16:a0.25:classes=ins_buzz+ambient_rain+human"`

## Results
Seed 2 (`fe_C_fast32h16_a0.25_s2_c-buzz-rain-human_wsd56000`) against seed 1 at the same budget, `tools/results.py` on the eval dirs:

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.670 | 0.692 | +0.022 | 0.017 | 34 |
| 53 | 0.641 | 0.638 | -0.003 | 0.013 | 32 |
| 1_11 | 0.586 | 0.600 | +0.014 | 0.011 | 32 |
| 1_143 | 0.567 | 0.569 | +0.002 | 0.022 | 26 |
| 1_37 | 0.597 | 0.583 | -0.014 | 0.018 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.612 → this 0.616 (+0.004 ± 0.007)
- inclusive (sensitivity), same thresholds: 0.505 → 0.511 (+0.006)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.730 | 0.773 | +0.043 | 603 |
| untagged | 0.616 | 0.624 | +0.008 | 7485 |
| background | 0.607 | 0.610 | +0.003 | 9412 |
| quiet | 0.167 | 0.174 | +0.007 | 2231 |
| faint | 0.000 | 0.000 | +0.000 | 35 |

± SD is eval sampling only; seed noise is not in it.

Step-budget curve, headline (seed 1 / seed 2): 7000 0.546 / 0.549, 14000 0.569 / 0.589, 28000 0.603 / 0.611, 56000 0.612 / 0.616. Speed at 200 s: 2.75x / 2.73x YAMNet. Both seeds are still rising at 56000, slowly (+0.009 and +0.005 over the last doubling).

Interpretation. The seeds agree at the judged budget: +0.004 ± 0.007, well inside the 0.01-0.02 seed noise, so the hypothesis holds (not falsified; the bar was a gap over 0.02). The curves are within 0.008 of each other at every budget except 14000 (+0.020), which closes by 28000. No fold moved beyond eval SD plus seed noise: `1_29` +0.022 ± 0.017 and `1_11` +0.014 ± 0.011 are each ~1.3 eval SDs before seed noise, so unsure; `53`, `1_143` and `1_37` are within their SDs. Tiers: `loud` +0.043 is the only one that stands out, on 603 frames; `untagged` +0.008, `background` +0.003, `quiet` +0.007 are flat. `faint` is 35 frames at zero for both.

Against the frontier. `fast32h16 a0.25` is 0.614 (two seeds) at 2.74x and stays the fastest rung-C point; nothing is at its speed or faster.

What this does to `fused-blocks`. `a0.25_d8_f2` (0.630 at 2.73x, one seed) was +0.018 ± 0.008 over seed 1; against seed 2 it is +0.014, against the two-seed mean +0.016. Still under the 0.02 bar and still one seed on the fused side, so its reading is unchanged: a marginal fastest point, about one seed-noise width above the separable trunk at the same speed. The tier part of that reading weakens: `fused-blocks` reported `loud` +0.036 over seed 1, and seed 2 alone moves `loud` by +0.043 with no architecture change, so that `loud` gain is inside seed-to-seed spread and should not be credited to the fused blocks. Its `untagged` +0.020 shrinks to about +0.012 against seed 2.

## Conclusion
Reproducible. `fast32h16:a0.25` at rung C is 0.614 (two seeds, 0.612 and 0.616) at 2.74x YAMNet, incl. quiet 0.508; the seeds differ by +0.004 ± 0.007 and by at most 0.020 along the budget curve. Tiers: `loud` +0.043 between seeds (603 frames), the rest within 0.008. The fast end of the frontier no longer rests on one seed. `fused-blocks`' `a0.25_d8_f2` is +0.016 over the two-seed mean, under the 0.02 bar as before, and its `loud` +0.036 is no larger than the seed-to-seed `loud` spread measured here, so only its small headline edge remains. Where to dig: `loud` is a 603-frame tier that moves 0.04 between seeds at this width; don't read tier deltas of that size on fast-end students from one seed. Both seeds still climb slowly at 56000, so a 112000 budget would be worth ~0.005-0.01 at most.
