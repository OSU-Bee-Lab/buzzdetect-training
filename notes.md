# shallow-wide
## Hypothesis
Two facts pull against each other on the distillation frontier: width is where students' sensitivity lives (fast32h16 a0.25 0.612 -> a0.375 0.639 -> a0.50 0.650 at 56k), and depth beyond 10 layers is free (`fast32-depth`: fast32 a0.50_d10 0.696 vs full depth 0.687). So spend the depth saved on width: fast32h16 a0.75_d8 (8 layers, 384 ch out; 6.2 MMACs vs a0.50 full's 5.9). Random-weight bench (bench_arch, 200 s, GPU): a0.75_d8@fast32h16 2.36x, a0.50@fast32h16 2.50x, a1.00_d8 2.20x, a0.75_d6 2.41x. Rung C, buzz+rain+human, seed 1, WSD to 56k with the stop rule. Matched control: fe_C_fast32h16_a0.50_s1_c-buzz-rain-human (0.650 @2.50x); depth-matched comparator: fast32h16 a0.50_d8 (`fast32h16-depth`). Gain: headline > 0.650 (beats every student at its speed or faster). Falsifier: a0.75_d8 <= a0.50_d8's headline (width does not help a shallow trunk) or <= 0.650.

## Results
| student | headline | incl. | speed | hit@K % |
|---|---|---|---|---|
| fast32h16 a0.50 (control) | 0.650 | 0.541 | 2.50x | |
| fast32h16 a0.50_d8 (depth-matched) | 0.646 | 0.536 | 2.64x | 63.9 |
| **fast32h16 a0.75_d8** | 0.656 | 0.544 | 2.46x | 64.7 |

Step-budget curve: 0.631 (7k), 0.630 (14k), 0.650 (28k), 0.656 (56k).

### fast32h16_a0.75_d8 vs fast32h16_a0.50
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.764 | 0.751 | -0.013 | 0.014 | 34 |
| 53 | 0.639 | 0.640 | +0.001 | 0.009 | 32 |
| 1_11 | 0.587 | 0.630 | +0.043 | 0.015 | 32 |
| 1_143 | 0.627 | 0.621 | -0.006 | 0.010 | 26 |
| 1_37 | 0.634 | 0.637 | +0.003 | 0.018 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.650 → this 0.656 (+0.006 ± 0.006)
- inclusive (sensitivity), same thresholds: 0.541 → 0.544 (+0.003)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.765 | 0.793 | +0.028 | 603 |
| untagged | 0.647 | 0.656 | +0.009 | 7485 |
| background | 0.664 | 0.654 | -0.010 | 9412 |
| quiet | 0.188 | 0.194 | +0.006 | 2231 |
| faint | 0.000 | 0.000 | +0.000 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.
### fast32h16_a0.75_d8 vs fast32h16_a0.50_d8
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.728 | 0.751 | +0.023 | 0.010 | 34 |
| 53 | 0.647 | 0.640 | -0.007 | 0.012 | 32 |
| 1_11 | 0.608 | 0.630 | +0.022 | 0.011 | 32 |
| 1_143 | 0.607 | 0.621 | +0.014 | 0.013 | 26 |
| 1_37 | 0.642 | 0.637 | -0.005 | 0.018 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.646 → this 0.656 (+0.010 ± 0.006)
- inclusive (sensitivity), same thresholds: 0.536 → 0.544 (+0.008)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.778 | 0.793 | +0.015 | 603 |
| untagged | 0.649 | 0.656 | +0.007 | 7485 |
| background | 0.642 | 0.654 | +0.012 | 9412 |
| quiet | 0.193 | 0.194 | +0.001 | 2231 |
| faint | 0.000 | 0.000 | +0.000 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Reading:
- **vs control (a0.50 full depth):** +0.006 ± 0.006 at 0.04x slower. Within eval SD and seed noise (~0.01-0.02). The one fold that moved is 1_11, this front end's hardest (0.587 -> 0.630, +0.043 ± 0.015); loud +0.028.
- **vs a0.50_d8 (depth-matched):** +0.010 ± 0.006, spread over 1_29 (+0.023 ± 0.010), 1_11 (+0.022 ± 0.011), 1_143 (+0.014 ± 0.013, unsure); all tiers but faint up. Width does help a shallow trunk, but costs 0.18x.
- **Frontier:** 0.656 @2.46x sits on the line between fast32 a0.50_d6 (0.664 @2.38x) and fast32h16 a0.50_d8 (0.646 @2.64x), ~0.002 below it. It beats the control on both axes only within noise.

## Conclusion
Neither gain nor falsifier is cleanly met: 0.656 > 0.650 but by +0.006 ± 0.006, and the student is 0.04x slower than the control. Spending saved depth on width trades along the frontier rather than lifting it: against the depth-matched a0.50_d8 it is +0.010 ± 0.006 for -0.18x, about the line's slope. Worth noting for the hard folds: 1_11 +0.043 ± 0.015 over the control, the largest single-fold move in the batch, on h16's weakest fold. Tiers: loud +0.028, others flat.
