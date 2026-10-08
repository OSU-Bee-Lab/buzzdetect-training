# yamnet-depth
## Hypothesis
`fast32-depth` and `fast32-d8d6` found fast32 a0.50's trunk can lose layers 11-14 at no sensitivity cost (d10 0.696 vs full 0.687) and layers 9-14 at none either (d8 0.685), but on fast32 the front end dominates wall time, so the cut bought little speed. On the YAMNet front end the trunk is a larger share of the time (yamnet a0.50 1.40x vs a0.25 1.62x: width alone moves speed by 0.22x), so a depth cut there should buy more. Student: yamnet a0.50 with its trunk cut to the depth (from a random-weight bench, see Changes) whose expected logged speed is >= a0.25's 1.62x. Rung C, buzz+rain+human (select), seed 1, WSD to 56k with the stop rule. Matched controls: fe_C_yamnet_a0.50_s1_select_c-buzz-rain-human (0.725 @1.40x, stop at 28k) and fe_C_yamnet_a0.25_s1_select_c-buzz-rain-human (0.707 @1.62x). Gain: headline > 0.707 at >= 1.62x (moves the frontier's high-sensitivity end). Falsifier: headline <= 0.707 at its speed, i.e. at the YAMNet front end depth costs what width does.

**Revised after the bench (before any training).** Random-weight bench (bench_arch, 200 s, GPU, x YAMNet; on this front end the bench matches logged speeds: a0.50 1.37 vs 1.40 logged, a0.25 1.62 vs 1.62): a0.50 1.37, _d12 1.42, _d10 1.46, _d8 1.50, _d6 1.55, a0.25 1.62, front end alone 1.78. No depth reaches a0.25's speed: the YAMNet front end caps the trunk's share as it does on fast32. The question becomes whether depth beats width as a speed lever here too (fast32h16-depth: weakly yes). Student: **yamnet a0.50_d8** (~1.50x; fast32's d8 cost nothing against full depth, d6 cost 0.03). Gain: headline above the yamnet a0.50 -> a0.25 line at its speed (0.725 @1.40x -> 0.707 @1.62x, i.e. > ~0.717 at 1.50x) by more than seed noise. Falsifier: <= 0.717, i.e. the cut loses as much per unit speed as width does.

## Results
| student | headline | incl. | speed (logged) | hit@K % |
|---|---|---|---|---|
| yamnet a0.50 (control, wsd28000) | 0.725 | 0.609 | 1.40x | 71.0 |
| yamnet a0.25 (control, wsd56000) | 0.707 | 0.594 | 1.62x | |
| **yamnet a0.50_d8** (wsd56000) | 0.735 | 0.618 | 1.52x | 69.9 |

Step-budget curve: 0.727 (7k), 0.731 (14k), 0.735 (28k), 0.735 (56k). Stop rule: hit@K gain +2.41, +1.11, +0.39, so it ran to the 56k cap. The recorded student is wsd56000, though 28k is equal on headline.

### a0.50_d8 vs a0.50 (full depth)
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.821 | 0.829 | +0.008 | 0.010 | 34 |
| 53 | 0.660 | 0.654 | -0.006 | 0.009 | 32 |
| 1_11 | 0.656 | 0.713 | +0.057 | 0.015 | 32 |
| 1_143 | 0.729 | 0.717 | -0.012 | 0.019 | 26 |
| 1_37 | 0.758 | 0.764 | +0.006 | 0.012 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.725 → this 0.735 (+0.010 ± 0.006)
- inclusive (sensitivity), same thresholds: 0.609 → 0.618 (+0.009)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.820 | 0.828 | +0.008 | 603 |
| untagged | 0.727 | 0.742 | +0.015 | 7485 |
| background | 0.706 | 0.704 | -0.002 | 9412 |
| quiet | 0.248 | 0.257 | +0.009 | 2231 |
| faint | 0.067 | 0.067 | +0.000 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

### a0.50_d8 vs a0.25
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.819 | 0.829 | +0.010 | 0.015 | 34 |
| 53 | 0.642 | 0.654 | +0.012 | 0.016 | 32 |
| 1_11 | 0.668 | 0.713 | +0.045 | 0.014 | 32 |
| 1_143 | 0.689 | 0.717 | +0.028 | 0.015 | 26 |
| 1_37 | 0.717 | 0.764 | +0.047 | 0.020 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.707 → this 0.735 (+0.028 ± 0.007)
- inclusive (sensitivity), same thresholds: 0.594 → 0.618 (+0.024)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.816 | 0.828 | +0.012 | 603 |
| untagged | 0.707 | 0.742 | +0.035 | 7485 |
| background | 0.696 | 0.704 | +0.008 | 9412 |
| quiet | 0.235 | 0.257 | +0.022 | 2231 |
| faint | 0.000 | 0.067 | +0.067 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

## Conclusion
**Gain, and stronger than the revised hypothesis asked for.** Cutting YAMNet a0.50's trunk to 8 layers gives 0.735 @1.52x. The bar was > ~0.717 at 1.50x (the a0.50 -> a0.25 width line). The student beats that bar by ~0.018, and it beats the full-depth a0.50 by +0.010 ± 0.006 while running 0.12x faster. It dominates a0.50 on both axes and is +0.028 ± 0.007 over a0.25, which is only 0.10x faster. So on the YAMNet front end, layers 9-14 cost speed and buy no sensitivity, the same as on fast32 (d8 0.685 vs full 0.687). Depth is the better speed lever on this front end, not width. On the frontier, yamnet a0.50_d8 is now the high-sensitivity point at ~1.5x.

Per fold (investigation only): against a0.50, 1_11 is +0.057 ± 0.015 and the other folds are within about one SD. Against a0.25, every fold moves up, and the largest moves are 1_11 and 1_37 (+0.045, +0.047). The single seed is the caveat: the margin over a0.50 (+0.010) is near seed noise (0.006-0.007 elsewhere in this era). The margin over the width line (~+0.018) and over a0.25 (+0.028) is not.

Next: a d6 cut on yamnet a0.50 (bench 1.55x; d6 cost 0.03 on fast32), or a0.75_d8 (as in shallow-wide), to see whether the saved depth is better spent on width at the YAMNet front end.
