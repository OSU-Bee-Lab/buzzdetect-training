# fast32h16-depth
## Hypothesis
`fast32-depth` showed trunk depth beyond 10 layers is free in sensitivity on fast32. fast32h16 (hop 16 ms, the fastest front end on the frontier: 2.93x alone) is where the trunk is the largest remaining share of wall time, so a depth cut there should buy speed at a smaller headline cost than narrowing the width does (a0.50 0.650 @2.51x -> a0.375 0.639 @2.63x -> a0.25 0.612 @2.75x). Students: fast32h16 a0.50 with the trunk cut (depths chosen from a random-weight speed bench, see Changes), rung C, buzz+rain+human, seed 1, WSD to 56k with the stop rule. Matched control: fe_C_fast32h16_a0.50_s1_c-buzz-rain-human (0.650 @2.50x at 56k). Gain: a student above the frontier line at its speed, e.g. > 0.639 at >= 2.63x or > 0.612 at >= 2.75x. Falsifier: the cut students lose as much headline per unit of speed as the width ladder does.

## Changes
`a0.50_d10`, `a0.50_d8`, `a0.50_d6` added to `distill_train.ARCHS` and `bench_arch.CANDIDATES` (shared commit with `fast32-d8d6`).
Depth chosen from a random-weight bench (bench_arch, 200 s, GPU, x YAMNet): fast32h16 a0.50 2.50, _d10 2.60, _d8 2.66, _d6 2.65, a0.25 2.75, front end alone 2.97. d10 would land behind a0.375's 2.63x and d6 buys nothing over d8, so the student is **fast32h16 a0.50_d8** alone (fe_C_fast32h16_a0.50_d8_s1_c-buzz-rain-human_wsd*).

## Results
| student | headline | incl. | speed | hit@K % |
|---|---|---|---|---|
| fast32h16 a0.50 (control) | 0.650 | 0.541 | 2.50x | |
| fast32h16 a0.375 (width neighbour) | 0.639 | 0.532 | 2.63x | |
| fast32h16 a0.25 (width neighbour) | 0.612 | 0.505 | 2.75x | |
| **fast32h16 a0.50_d8** | 0.646 | 0.536 | 2.64x | 63.9 |

Step-budget curve: 0.624 (7k), 0.640 (14k), 0.648 (28k), 0.646 (56k): flat from 28k.

### fast32h16_a0.50_d8 vs fast32h16_a0.50
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.764 | 0.728 | -0.036 | 0.015 | 34 |
| 53 | 0.639 | 0.647 | +0.008 | 0.014 | 32 |
| 1_11 | 0.587 | 0.608 | +0.021 | 0.010 | 32 |
| 1_143 | 0.627 | 0.607 | -0.020 | 0.013 | 26 |
| 1_37 | 0.634 | 0.642 | +0.008 | 0.022 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.650 → this 0.646 (-0.004 ± 0.007)
- inclusive (sensitivity), same thresholds: 0.541 → 0.536 (-0.005)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.765 | 0.778 | +0.013 | 603 |
| untagged | 0.647 | 0.649 | +0.002 | 7485 |
| background | 0.664 | 0.642 | -0.022 | 9412 |
| quiet | 0.188 | 0.193 | +0.005 | 2231 |
| faint | 0.000 | 0.000 | +0.000 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Reading:
- **vs control:** -0.004 ± 0.007 headline for +0.14x speed. Folds moved both ways: 1_29 -0.036 ± 0.015 (the easiest fold), 1_143 -0.020 ± 0.013 (unsure), 1_11 +0.021 ± 0.010 (a hard fold for this front end). Loud +0.013, background -0.022: unsure, within tier noise.
- **vs the width ladder:** a0.375 buys +0.13x for -0.011; the depth cut buys +0.14x for -0.004. At matched speed (2.64x vs 2.63x) d8 is +0.007 over a0.375, an unpaired comparison inside seed noise (~0.01-0.02).
- **Frontier:** new point at 2.64x, displacing a0.375 (0.639 @2.63x) by a margin within noise.

## Conclusion
Weak win. On fast32h16, cutting the trunk to 8 layers is a cheaper speed lever than narrowing it: +0.14x for -0.004 ± 0.007, vs width's +0.13x for -0.011. It takes a0.375's place on the frontier (0.646 vs 0.639 at ~2.63x), but by less than one seed's noise; a second seed would settle whether depth beats width here. Next on this axis: a0.375_d8 or a0.25_d8 for the >2.7x end. Tiers flat (loud +0.013, background -0.022, both unsure); the fold changes cancel (1_29 -0.036 ± 0.015, 1_11 +0.021 ± 0.010).
