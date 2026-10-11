# fast32-d8-repeat
## Hypothesis
Distillation arm, repeat rule. `fe_C_fast32_a0.50_d8_s1_c-buzz-rain-human_wsd56000` (0.685 at 2.34x YAMNet) is +0.021 over every rung-C student at its speed or faster, on one seed; `ladder_record.py frontier` lists it as awaiting a repeat. Headline seed noise on a student is ~0.01-0.02, so the gain counts only if seed 2 of the identical run (same teacher, rung, WSD budgets, front end, width, depth, classes) lands within that noise of seed 1 and still above everything at its speed or faster. Falsified if seed 2 falls back onto the existing frontier (gain <= 0.02 over `fast32 a0.50_d6`, 0.664 at 2.38x).

Matched control: the seed-1 row itself, at the same budget.

## Changes
None. `ladder_record.py frontier`'s repeat command, verbatim:
`05_distill/main.py --rung C --seed 2 --wsd 7000,14000,28000,56000 --runs "fast32:a0.50_d8:classes=ins_buzz+ambient_rain+human"`

## Results
Seed 2 (`fe_C_fast32_a0.50_d8_s2_c-buzz-rain-human_wsd56000`) against seed 1 at the same budget, `tools/results.py` on the eval dirs:

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.764 | 0.744 | -0.020 | 0.016 | 34 |
| 53 | 0.656 | 0.643 | -0.013 | 0.010 | 32 |
| 1_11 | 0.655 | 0.642 | -0.013 | 0.011 | 32 |
| 1_143 | 0.645 | 0.647 | +0.002 | 0.013 | 26 |
| 1_37 | 0.703 | 0.713 | +0.010 | 0.013 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.685 → this 0.678 (-0.007 ± 0.006)
- inclusive (sensitivity), same thresholds: 0.569 → 0.565 (-0.004)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.788 | 0.773 | -0.015 | 603 |
| untagged | 0.688 | 0.682 | -0.006 | 7485 |
| background | 0.670 | 0.652 | -0.018 | 9412 |
| quiet | 0.210 | 0.217 | +0.007 | 2231 |
| faint | 0.067 | 0.133 | +0.066 | 35 |

± SD is eval sampling only; seed noise is not in it.

Step-budget curve, headline (seed 1 / seed 2): 7000 0.669 / 0.663, 14000 0.675 / 0.676, 28000 0.686 / 0.682, 56000 0.685 / 0.678. Speed at 200 s: 2.34x / 2.33x YAMNet. Both seeds flatten after 28000 (seed 2 dips -0.004 at 56000 while hit@K still rises +1.16).

Interpretation. The seeds agree at the judged budget: -0.007 ± 0.006, inside the 0.01-0.02 seed noise, and within 0.007 of each other at every budget. No fold moved beyond what eval SD plus seed noise allows: `1_29` -0.020 ± 0.016, `53` -0.013 ± 0.010 and `1_11` -0.013 ± 0.011 are each about 1.2 eval SDs before seed noise, so unsure; `1_143` and `1_37` are within their SDs. Tiers: `background` -0.018 and `loud` -0.015 lean down together with the `1_29`/`53` folds that hold them, `untagged` -0.006, `quiet` +0.007; none is outside seed noise. `faint` +0.066 is 35 frames (two frames' worth), not a reading.

Against the frontier. The best rung-C student at 2.33x or faster is `fast32 a0.50_d6` (0.664 at 2.38x, one seed). Seed 1 was +0.021 over it, which triggered the repeat; seed 2 is +0.014; the two-seed mean is 0.682 at 2.34x, +0.018. By the hypothesis as written (gain > 0.02 on the repeat), the step does not confirm: it is a gain of about 0.015-0.02 over a single-seed neighbour, the size of seed noise. It is still non-dominated: nothing at its speed or faster reaches it, and the next student up, `fast32 a0.375` (0.675 at 2.27x), is slower and 0.007 lower, i.e. the same point within noise, as `fast32h16-d8-repeat` found for the h16 pair.

## Conclusion
Reproducible, not a confirmed frontier step. `fast32:a0.50_d8` at rung C is 0.682 (two seeds, 0.685 and 0.678) at 2.34x YAMNet, incl. quiet 0.567; the seeds differ by -0.007 ± 0.006 and no tier moved outside seed noise (`background` -0.018, `loud` -0.015, `untagged` -0.006, `quiet` +0.007). Seed 2 is +0.014 over `fast32 a0.50_d6` (0.664 at 2.38x), under the repeat rule's 0.02, so seed 1's +0.021 was partly a high draw; the two-seed margin is +0.018 against a one-seed neighbour. Read: on fast32, depth 8 at width 0.50 holds the ~0.68 at 2.3x point together with plain width 0.375 (0.675 at 2.27x); cutting depth to 8 buys ~0.2x of speed over full-depth a0.50 (0.687 at 2.15x) for no measurable headline cost, which is the useful result. Where to dig: the d6 -> d8 step (0.664 -> 0.682) is the only place depth shows a slope here, and d6 has one seed.
