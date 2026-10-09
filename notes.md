# yamnet-a025-d8
## Hypothesis
The YAMNet front end caps a student near 1.78x logged (front end alone). Between yamnet a0.50_d8 (0.735 @1.52x) and fast32 a0.50_d10 (0.696 @2.28x) the frontier has no student; fast32 a0.50_d12 (0.697 @2.21x) is the best at >= 1.75x. Random-weight bench (bench_arch, 20 s, GPU, one run, x YAMNet; within-bench ratios only): a0.50 1.50, a0.375 1.67, a0.25 1.88, a0.50_d8 1.76, a0.50_d6 1.82, a0.375_d8 1.88, a0.25_d8 2.00, a0.75_d8 1.39, a0.75_d6 1.62, a0.375_d6 1.92, a1.00_d6 1.48. Logged/bench ratio on this front end: a0.50 0.93, a0.25 0.86, a0.50_d8 0.86. a0.25_d8 is the fastest d8 candidate (2.00 bench, ~1.72-1.86x logged), close to the front-end cap. Student: yamnet a0.25_d8, seed 1. Rung C, buzz+rain+human (select), WSD to 56k with the stop rule (--wsd-max 56000 --wsd-stop 1.3), teacher v4-ft-ps-e60-moderate. Matched controls: yamnet a0.25 full depth (0.707 @1.62x, depth only) and yamnet a0.50_d8 (width only). Gain: headline > 0.697 (best at its speed or faster) by more than seed noise, i.e. the YAMNet front end beats fast32 at ~1.75x. Falsifier: <= 0.697, i.e. at that speed a fast front end is the better buy.

## Changes
No code change on this branch. Student `yamnet:a0.25_d8:select:classes=ins_buzz+ambient_rain+human`, seed 1, run in batch 26's single chain from the `yamnet-d8-repeat` worktree (the branch with the `parse_arch` fix), `--rung C --wsd-max 56000 --wsd-stop 1.3`. The stop rule did not fire; the curve ran to 56k.

## Results
WSD curve (headline @ logged 200 s speed): 7k 0.676 @1.69x, 14k 0.691 @1.68x, 28k 0.705 @1.69x, 56k 0.713 @1.69x. Judged point: wsd56000, **0.713 @1.69x**. Still rising at 56k (+0.008 over 28k), unlike a0.375_d8 and a0.50_d8, which flattened by 28k.

Comparisons, in order: (1) yamnet a0.25 full depth s1 wsd56000 (0.707 @1.62x, depth only); (2) yamnet a0.50_d8 s1 wsd56000 (0.735 @1.52x, width only); (3) yamnet a0.375_d8 s1 wsd56000 (0.716 @1.58x).

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.819 | 0.799 | -0.020 | 0.020 | 34 |
| 53 | 0.642 | 0.651 | +0.009 | 0.015 | 32 |
| 1_11 | 0.668 | 0.676 | +0.008 | 0.012 | 32 |
| 1_143 | 0.689 | 0.690 | +0.001 | 0.015 | 26 |
| 1_37 | 0.717 | 0.749 | +0.032 | 0.023 | 15 |
- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.707 → this 0.713 (+0.006 ± 0.008)
- inclusive (sensitivity), same thresholds: 0.594 → 0.600 (+0.006)
- folds that missed fpr 0.005: none
| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.816 | 0.811 | -0.005 | 603 |
| untagged | 0.707 | 0.722 | +0.015 | 7485 |
| background | 0.696 | 0.682 | -0.014 | 9412 |
| quiet | 0.235 | 0.249 | +0.014 | 2231 |
| faint | 0.000 | 0.067 | +0.067 | 35 |
± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.
======
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.829 | 0.799 | -0.030 | 0.015 | 34 |
| 53 | 0.654 | 0.651 | -0.003 | 0.014 | 32 |
| 1_11 | 0.713 | 0.676 | -0.037 | 0.011 | 32 |
| 1_143 | 0.717 | 0.690 | -0.027 | 0.016 | 26 |
| 1_37 | 0.764 | 0.749 | -0.015 | 0.014 | 15 |
- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.735 → this 0.713 (-0.022 ± 0.006)
- inclusive (sensitivity), same thresholds: 0.618 → 0.600 (-0.018)
- folds that missed fpr 0.005: none
| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.828 | 0.811 | -0.017 | 603 |
| untagged | 0.742 | 0.722 | -0.020 | 7485 |
| background | 0.704 | 0.682 | -0.022 | 9412 |
| quiet | 0.257 | 0.249 | -0.008 | 2231 |
| faint | 0.067 | 0.067 | +0.000 | 35 |
± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.
======
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.793 | 0.799 | +0.006 | 0.016 | 34 |
| 53 | 0.647 | 0.651 | +0.004 | 0.013 | 32 |
| 1_11 | 0.680 | 0.676 | -0.004 | 0.013 | 32 |
| 1_143 | 0.704 | 0.690 | -0.014 | 0.019 | 26 |
| 1_37 | 0.757 | 0.749 | -0.008 | 0.015 | 15 |
- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.716 → this 0.713 (-0.003 ± 0.007)
- inclusive (sensitivity), same thresholds: 0.606 → 0.600 (-0.006)
- folds that missed fpr 0.005: none
| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.832 | 0.811 | -0.021 | 603 |
| untagged | 0.724 | 0.722 | -0.002 | 7485 |
| background | 0.676 | 0.682 | +0.006 | 9412 |
| quiet | 0.261 | 0.249 | -0.012 | 2231 |
| faint | 0.067 | 0.067 | +0.000 | 35 |
± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

(1) Depth cut at a0.25: +0.006 ± 0.008 eval (~±0.012 with seed noise), unsure, while 0.07x faster (1.62 → 1.69x). Per fold all within SD except 1_37 +0.032 ± 0.023 (weak); 1_29 -0.020 ± 0.020 (unsure). Tiers: untagged +0.015, quiet +0.014, background -0.014. So d8 is at least free at a0.25 as at a0.50.
(2) Narrowing a0.50_d8 to a0.25_d8: -0.022 ± 0.006 (vs the two-seed a0.50_d8 mean 0.730, ~-0.017) for +0.17x. Losses on 1_11 (-0.037 ± 0.011), 1_29 (-0.030 ± 0.015), 1_143 (-0.027 ± 0.016, weak); every tier down ~0.02.
(3) Against a0.375_d8: -0.003 ± 0.007 for +0.11x; nothing outside SD. a0.375_d8 buys nothing a0.25_d8 doesn't.

Speed: bench 2.00 → logged 1.69x (ratio 0.845), the low end of the predicted 1.72-1.86x. It did not reach the ~1.75x where the hypothesis put it.

## Conclusion
Partial gain, not the one stated. The hypothesis' test (beat fast32 a0.50_d12's 0.697 at ~1.75x) is not met as posed: the student only reached 1.69x, so it does not reach the fast32 region (0.697 @2.21x is still the only buy above ~1.75x; twofast32 0.691 @1.75x). What it does: 0.713 @1.69x dominates yamnet a0.25 (0.707 @1.62x), faster and level within seed noise, and makes a0.375_d8 (0.716 @1.58x) redundant. New frontier point between a0.50_d8 (0.730 two-seed @1.51x) and the fast32 line. The d8 width line is a0.50 0.730 @1.51 → a0.375 0.716 @1.58 → a0.25 0.713 @1.69: below a0.50 width is close to free in headline, and the 0.50→0.375 step is where the cost lands (1_11, 1_29, background tier). The YAMNet front end still caps near 1.7-1.8x, so the gap 1.75-2.2x stays unfilled. Curve still rising at 56k; a longer budget might add a little.
