# fast32-d8d6
## Hypothesis
`fast32-depth` found that cutting fast32 a0.50's trunk to 12 or 10 layers costs no sensitivity (0.697 / 0.696 vs 0.687 full depth at 56k) but buys little speed (2.15x -> 2.28x). If depth is genuinely not where this student's sensitivity lives, the cut can go further: a0.50_d8 (layers 9-14 removed: ends two layers into the 512-wide block, 256 ch at a0.50) and a0.50_d6 (layers 7-14 removed: ends before the 512 block, 128 ch at a0.50, one stride-2 fewer). Rung C, buzz+rain+human, seed 1, WSD to 56k with the stop rule. Matched controls: fe_C_fast32_a0.50_s1_c-buzz-rain-human (0.687 @2.15x) and fe_C_fast32_a0.50_d10 (0.696 @2.28x). Gain: either student above the frontier line at its speed (fast32h16 a0.50 0.650 @2.51x; fast32h16 a0.375 0.639 @2.63x). Falsifier: headline falls below d10's by more than ~0.03 with < 0.1x of speed gained, or speed stays under 2.4x (front end alone is 2.70x, so the ceiling is near).

## Changes
`a0.50_d10`, `a0.50_d8`, `a0.50_d6` added to `distill_train.ARCHS` and `bench_arch.CANDIDATES`.
Random-weight bench (bench_arch, 200 s, GPU, x YAMNet; fast32 rows read ~0.6x low against trained students' speed_200, so only ratios count): a0.50 1.55, _d10 1.61, _d8 1.64, _d6 1.67, front end alone 2.72. Expected logged speeds by ratio: d8 ~2.3x, d6 ~2.35x.
Run as the first part of one chained job (`chain.sh` in this worktree) that then runs `fast32h16-depth` and `shallow-wide` from their own worktrees.

## Results
Students at 56k (WSD, stop rule never fired; logged speed is x YAMNet at 200 s, GPU):

| student | headline | incl. | speed | hit@K % |
|---|---|---|---|---|
| fast32 a0.50 (full, control) | 0.687 | 0.571 | 2.15x | |
| fast32 a0.50_d10 (control) | 0.696 | 0.577 | 2.28x | |
| **fast32 a0.50_d8** | 0.685 | 0.569 | 2.34x | 66.0 |
| **fast32 a0.50_d6** | 0.664 | 0.553 | 2.38x | 62.3 |

Step-budget curves: d8 plateaued at 28k (0.686 -> 0.685 at 56k); d6 was still climbing (0.641 at 14k, 0.653 at 28k, 0.664 at 56k), so its gap to d8/d10 may be partly a slower-learning trunk at the same budget.

### fast32_a0.50_d8 vs fast32_a0.50_d10
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.783 | 0.764 | -0.019 | 0.014 | 34 |
| 53 | 0.659 | 0.656 | -0.003 | 0.010 | 32 |
| 1_11 | 0.668 | 0.655 | -0.013 | 0.014 | 32 |
| 1_143 | 0.658 | 0.645 | -0.013 | 0.011 | 26 |
| 1_37 | 0.711 | 0.703 | -0.008 | 0.013 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.696 → this 0.685 (-0.011 ± 0.006)
- inclusive (sensitivity), same thresholds: 0.577 → 0.569 (-0.008)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.799 | 0.788 | -0.011 | 603 |
| untagged | 0.696 | 0.688 | -0.008 | 7485 |
| background | 0.686 | 0.670 | -0.016 | 9412 |
| quiet | 0.210 | 0.210 | +0.000 | 2231 |
| faint | 0.133 | 0.067 | -0.066 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.
### fast32_a0.50_d8 vs fast32_a0.50
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.760 | 0.764 | +0.004 | 0.014 | 34 |
| 53 | 0.658 | 0.656 | -0.002 | 0.010 | 32 |
| 1_11 | 0.668 | 0.655 | -0.013 | 0.019 | 32 |
| 1_143 | 0.660 | 0.645 | -0.015 | 0.011 | 26 |
| 1_37 | 0.688 | 0.703 | +0.015 | 0.013 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.687 → this 0.685 (-0.002 ± 0.006)
- inclusive (sensitivity), same thresholds: 0.571 → 0.569 (-0.002)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.794 | 0.788 | -0.006 | 603 |
| untagged | 0.690 | 0.688 | -0.002 | 7485 |
| background | 0.670 | 0.670 | +0.000 | 9412 |
| quiet | 0.212 | 0.210 | -0.002 | 2231 |
| faint | 0.133 | 0.067 | -0.066 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.
### fast32_a0.50_d6 vs fast32_a0.50_d10
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.783 | 0.758 | -0.025 | 0.018 | 34 |
| 53 | 0.659 | 0.663 | +0.004 | 0.011 | 32 |
| 1_11 | 0.668 | 0.641 | -0.027 | 0.015 | 32 |
| 1_143 | 0.658 | 0.590 | -0.068 | 0.016 | 26 |
| 1_37 | 0.711 | 0.669 | -0.042 | 0.016 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.696 → this 0.664 (-0.032 ± 0.007)
- inclusive (sensitivity), same thresholds: 0.577 → 0.553 (-0.024)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.799 | 0.796 | -0.003 | 603 |
| untagged | 0.696 | 0.665 | -0.031 | 7485 |
| background | 0.686 | 0.672 | -0.014 | 9412 |
| quiet | 0.210 | 0.203 | -0.007 | 2231 |
| faint | 0.133 | 0.133 | +0.000 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Reading:
- **d8 vs full depth:** -0.002 ± 0.006, no fold moved past its SD; tiers flat (faint is 35 frames). Six trunk layers cut at no measurable sensitivity cost, +0.19x speed.
- **d8 vs d10:** -0.011 ± 0.006 for +0.06x. Every fold down by about its own SD (1_29 -0.019 ± 0.014, 1_143 -0.013 ± 0.011): small, consistent, unsure per fold. Seed noise on the headline is ~0.01-0.02, so d8 ≈ d10 within one draw.
- **d6 vs d10:** -0.032 ± 0.007 for +0.10x. The loss is concentrated on the hardest fold, 1_143 (-0.068 ± 0.016), and 1_37 (-0.042 ± 0.016); 1_11 -0.027 ± 0.015. Untagged tier carries it (-0.031); loud and quiet flat.
- **Frontier:** both sit on it (nothing faster scores higher): d8 0.685 @2.34x, d6 0.664 @2.38x, ahead of fast32h16's 0.656 @2.46x. d8 lies on the d10 -> fast32h16 a0.50 line (+0.002 above it at 2.34x); d6 is ~0.01 below it.

## Conclusion
Falsifier met on speed: neither cut reaches 2.4x. On fast32 the trunk is exhausted as a speed lever: d10 -> d8 -> d6 buys +0.06x and +0.04x while the front end alone runs at ~2.7x. d8 is free against full depth (-0.002 ± 0.006) and only ~0.01 below d10; d6 costs 0.032 ± 0.007, mostly on hard fold 1_143 (-0.068 ± 0.016), so it is not worth its 0.04x. d10 remains the best fast32 point; further fast32 speed has to come from the front end (FFT/hop), not the trunk. Tiers: flat for d8; d6's loss is in untagged, not quiet.
