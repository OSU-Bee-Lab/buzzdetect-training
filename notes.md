# yamnet-a0375-d8
## Hypothesis
`yamnet-depth` found that on the YAMNet front end cutting a0.50's trunk to 8 layers costs nothing (0.735 @1.52x vs full 0.725 @1.40x): depth is a better speed lever than width. Does the d8 cut also hold at a narrower width, moving the 1.6x point? Random-weight bench (bench_arch, 20 s, GPU, one run, x YAMNet; within-bench ratios only): a0.50 1.50, a0.375 1.67, a0.25 1.88, a0.50_d8 1.76, a0.50_d6 1.82, a0.375_d8 1.88, a0.25_d8 2.00, a0.75_d8 1.39, a0.75_d6 1.62, a0.375_d6 1.92, a1.00_d6 1.48. Logged/bench ratio on this front end: a0.50 0.93, a0.25 0.86, a0.50_d8 0.86. a0.375_d8 benches level with full-depth a0.25 (1.88), so expect ~1.62x logged. Student: yamnet a0.375_d8, seed 1. Rung C, buzz+rain+human (select), WSD to 56k with the stop rule (--wsd-max 56000 --wsd-stop 1.3), teacher v4-ft-ps-e60-moderate. Matched controls: yamnet a0.50_d8 (0.735 @1.52x, width only) and yamnet a0.25 (0.707 @1.62x, the frontier point at that speed). Gain: headline > 0.707 at >= ~1.62x by more than seed noise (~0.02), i.e. it replaces a0.25 on the frontier. Falsifier: <= 0.707 at its speed, i.e. narrowing a d8 trunk costs at least what the full-depth width line does.

## Changes
No code change on this branch. Student `yamnet:a0.375_d8:select:classes=ins_buzz+ambient_rain+human`, seed 1, run in batch 26's single chain from the `yamnet-d8-repeat` worktree (the branch with the `parse_arch` fix), `--rung C --wsd-max 56000 --wsd-stop 1.3`. The stop rule did not fire; the curve ran to 56k.

## Results
WSD curve (headline @ logged 200 s speed): 7k 0.712 @1.58x, 14k 0.711 @1.58x, 28k 0.720 @1.60x, 56k 0.716 @1.58x. Judged point: wsd56000, **0.716 @1.58x**.

Control 1 = yamnet a0.50_d8 s1 wsd56000 (0.735 @1.52x); control 2 = yamnet a0.25 s1 wsd56000 (0.707 @1.62x).

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.829 | 0.793 | -0.036 | 0.016 | 34 |
| 53 | 0.654 | 0.647 | -0.007 | 0.014 | 32 |
| 1_11 | 0.713 | 0.680 | -0.033 | 0.013 | 32 |
| 1_143 | 0.717 | 0.704 | -0.013 | 0.013 | 26 |
| 1_37 | 0.764 | 0.757 | -0.007 | 0.010 | 15 |
- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.735 → this 0.716 (-0.019 ± 0.006)
- inclusive (sensitivity), same thresholds: 0.618 → 0.606 (-0.012)
- folds that missed fpr 0.005: none
| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.828 | 0.832 | +0.004 | 603 |
| untagged | 0.742 | 0.724 | -0.018 | 7485 |
| background | 0.704 | 0.676 | -0.028 | 9412 |
| quiet | 0.257 | 0.261 | +0.004 | 2231 |
| faint | 0.067 | 0.067 | +0.000 | 35 |
± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.
======
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.819 | 0.793 | -0.026 | 0.018 | 34 |
| 53 | 0.642 | 0.647 | +0.005 | 0.014 | 32 |
| 1_11 | 0.668 | 0.680 | +0.012 | 0.012 | 32 |
| 1_143 | 0.689 | 0.704 | +0.015 | 0.015 | 26 |
| 1_37 | 0.717 | 0.757 | +0.040 | 0.019 | 15 |
- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.707 → this 0.716 (+0.009 ± 0.007)
- inclusive (sensitivity), same thresholds: 0.594 → 0.606 (+0.012)
- folds that missed fpr 0.005: none
| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.816 | 0.832 | +0.016 | 603 |
| untagged | 0.707 | 0.724 | +0.017 | 7485 |
| background | 0.696 | 0.676 | -0.020 | 9412 |
| quiet | 0.235 | 0.261 | +0.026 | 2231 |
| faint | 0.000 | 0.067 | +0.067 | 35 |
± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Against a0.50_d8 (width only): -0.019 ± 0.006 eval, ~±0.010 with seed noise (the two a0.50_d8 seeds differ by 0.010; against their 0.730 mean the cost is ~-0.014). Losses concentrate on 1_29 (-0.036 ± 0.016) and 1_11 (-0.033 ± 0.013), both outside eval SD; 53, 1_143 and 1_37 are within it (unsure). Tiers: background (-0.028) and untagged (-0.018) carry the loss; loud and quiet unchanged. Speed gain is only +0.06x (1.52 → 1.58).

Against a0.25 full depth (the frontier point near this speed): +0.009 ± 0.007 eval (~±0.012 with seed noise), but 0.04x slower. Per fold: 1_37 +0.040 ± 0.019 (weak), 1_143 +0.015 ± 0.015 and 1_11 +0.012 ± 0.012 (unsure), 1_29 -0.026 ± 0.018 (weak loss). Quiet +0.026, background -0.020.

Speed came in below the bench prediction: bench 1.88 predicted ~1.62x logged; measured 1.58x (logged/bench 0.84, in line with a0.50_d8's 0.86).

## Conclusion
Neither gain nor falsifier: 0.716 @1.58x sits on the line between a0.50_d8 (two-seed 0.730 @1.51x) and a0.25 (0.707 @1.62x); linear interpolation at 1.58x gives ~0.715. It is slower than a0.25, so it does not replace it, and the +0.009 is within seed noise. Narrowing a d8 trunk from 0.50 to 0.375 costs ~0.014-0.019 for +0.06x, a worse exchange rate than d8 itself bought; the loss is in background/untagged tiers and folds 1_29 and 1_11. Width below 0.50 at d8 looks like a flat trade along the frontier, not a new point beyond it. a0.25_d8 (next in the chain) tests the faster end.
