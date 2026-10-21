# pauc-l01
## Hypothesis
IDEAS #33's remaining rung. `pauc-l03` (lambda 0.3: squared hinge, margin 1, between each `ins_buzz` positive's logit and the top 5% of the batch's negative logits) lost -0.091 +- 0.015 vs ps-depth8-r3, most on 1_114/1_150/1_95, with no logit collapse. At lambda 0.1 the BCE dominates more; if the loss at 0.3 was a dose overshoot, 0.1 lands at or above the three-draw depth8 mean (0.507) with hard folds up. Falsifier: monotone decline with dose (0.1 below 0.507, between it and pauc-l03's 0.406), as `pairwise-rank` showed; then the low-FPR-loss family closes. Watch each fold's `buzz logit SD` for collapse.
## Changes
None to code beyond exp/pauc-l03's `TRAIN_PAUC_LAMBDA` (patched in). Run: `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 TRAIN_PAUC_LAMBDA=0.1`, embedder `yamnet_trunk_pitchshift_depth8`, 30 epochs, medium.
## Results
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.422 | -0.064 | 0.044 | 32 |
| 53 | 0.550 | 0.535 | -0.015 | 0.029 | 28 |
| 1_11 | 0.573 | 0.554 | -0.019 | 0.033 | 26 |
| 1_143 | 0.586 | 0.608 | +0.022 | 0.047 | 22 |
| 1_150 | 0.435 | 0.296 | -0.139 | 0.041 | 21 |
| 1_95 | 0.284 | 0.253 | -0.031 | 0.030 | 46 |
| 1_37 | 0.648 | 0.557 | -0.091 | 0.051 | 14 |
| 1_114 | 0.414 | 0.179 | -0.235 | 0.049 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.426 (-0.071 ± 0.015)
- inclusive (sensitivity), same thresholds: 0.411 → 0.346 (-0.065)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.844 | -0.012 | 122 |
| untagged | 0.524 | 0.453 | -0.071 | 2418 |
| background | 0.406 | 0.363 | -0.043 | 1874 |
| quiet | 0.130 | 0.120 | -0.010 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Buzz logit SD per fold: 1.37, 1.13, 1.08, 0.89, 0.87, 0.95, 0.95, 1.03. No collapse.

Headline: 0.426 vs matched control ps-depth8-r3 0.497, **-0.071 ± 0.015** (eval sampling; ~±0.02 with training noise); -0.081 vs the depth8 three-draw mean 0.507. A real loss.

Against pauc-l03 (λ=0.3, 0.406): +0.020 ± 0.011, so λ=0.1 sits between the control and λ=0.3: the loss shrinks with dose but stays well below the control. Per fold vs pauc-l03: 1_95 +0.090 ± 0.029, 53 +0.059 ± 0.034, 1_29 +0.043 ± 0.025, 1_143 +0.022 ± 0.033 (unsure), 1_11 +0.020 ± 0.028 (unsure), 1_150 +0.000 ± 0.031, 1_37 -0.011 ± 0.026 (unsure), 1_114 -0.065 ± 0.041.

Per fold vs ps-depth8-r3: 1_114 -0.235 ± 0.049 and 1_150 -0.139 ± 0.041 are large losses on hard folds; 1_37 -0.091 ± 0.051 and 1_29 -0.064 ± 0.044 lean down; 1_95 -0.031 ± 0.030, 1_11 -0.019 ± 0.033, 53 -0.015 ± 0.029 (unsure); 1_143 +0.022 ± 0.047 (unsure). 1_114 is not dose-monotone: worse at λ=0.1 than at 0.3.

Tiers: untagged -0.071, background -0.043, loud -0.012, quiet -0.010 (flat).

## Conclusion
Falsified, and the idea's own falsifier is met: λ=0.1 gives 0.426, between the control (0.497 / three-draw 0.507) and λ=0.3 (0.406), so the decline is monotone in dose. Even a light pAUC term costs -0.071 ± 0.015, with the largest losses on hard folds 1_114 (-0.235) and 1_150 (-0.139). No logit collapse (SD 0.87-1.37). With pairwise-rank, tail-loss and mech-margin, this closes the low-FPR-loss family; IDEAS #33 deleted.
