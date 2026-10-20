# ps-depth8-e15
## Hypothesis
`ps-depth8-e60` showed depth8 overfits past 30 epochs (-0.060 ± 0.015 vs ps-depth8-r3), and both its and r3's per-fold held-out val_loss curves bottom out at epochs 2-19 (median ~9), at constant LR. If val_loss tracks sensitivity here, a 15-epoch budget sits nearer the minimum than 30 and lifts the headline, with the gain in `untagged`/`loud` and on the hard folds that lost most at 60 (1_114, 1_95, 1_150). It also halves fit time and sets the epoch budget for the final `moderate` run on depth8.

Prediction: headline above ps-depth8-r3 (0.497) and above the three-draw mean (0.507) by ≥0.02. Falsified if within ~0.02 of the three-draw mean (the 15-30 range is flat) or below it (undertrained: val_loss and sens@fpr disagree under 30).

Control: `ps-depth8-r3` (same recipe, 30 epochs, 2026-10-10); other draws `ps-depth8` 0.515, `ps-depth8-repeat` 0.509. Dose ladder with `ps-depth8-e60` (0.437).

## Changes
None in code. `--epochs 15` instead of 30; otherwise ps-depth8-r3's run (`TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, `--embedder yamnet_trunk_pitchshift_depth8`, medium, translation general).

## Results
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.476 | -0.010 | 0.037 | 32 |
| 53 | 0.550 | 0.560 | +0.010 | 0.028 | 28 |
| 1_11 | 0.573 | 0.571 | -0.002 | 0.030 | 26 |
| 1_143 | 0.586 | 0.595 | +0.009 | 0.035 | 22 |
| 1_150 | 0.435 | 0.303 | -0.132 | 0.042 | 21 |
| 1_95 | 0.284 | 0.246 | -0.038 | 0.025 | 46 |
| 1_37 | 0.648 | 0.584 | -0.064 | 0.042 | 14 |
| 1_114 | 0.414 | 0.331 | -0.083 | 0.054 | 28 |
- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.458 (-0.039 ± 0.013)
- inclusive (sensitivity), same thresholds: 0.411 → 0.381 (-0.030)
- folds that missed fpr 0.005: none
| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.811 | -0.045 | 122 |
| untagged | 0.524 | 0.483 | -0.041 | 2418 |
| background | 0.406 | 0.408 | +0.002 | 1874 |
| quiet | 0.130 | 0.138 | +0.008 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

Buzz logit SD 0.53-1.23 on every fold (no collapse). Final held-out val_loss is *lower* than r3's (30 epochs) on 7 of 8 folds (1_29 0.668 vs 0.690, 53 0.652 vs 0.662, 1_37 0.584 vs 0.599, 1_143 0.610 vs 0.621, 1_95 0.740 vs 0.748; 1_150 0.675 vs 0.662 the exception), yet sensitivity is down.

Headline -0.039 ± 0.013 vs ps-depth8-r3 (eval SD; ~3 SD, past the ~0.014 combined single-run SD), -0.049 vs the three-draw mean 0.507, below all five plain depth8 draws (0.464-0.515). The loss is on the hard folds and 1_37: 1_150 -0.132 ± 0.042, 1_114 -0.083 ± 0.054, 1_37 -0.064 ± 0.042, 1_95 -0.038 ± 0.025 (each near or past its eval SD, but identical runs scatter up to ~0.1 per fold, so each alone is unsure); the rich folds 1_29, 53, 1_11, 1_143 are flat (within ±0.010). Tiers: untagged -0.041, loud -0.045 (122 frames), background +0.002, quiet +0.008: isolated detections lose; the sustained drones are already fitted by epoch 15.

Dose ladder at depth8, same recipe: 15 → 0.458, 30 → 0.497 (draws 0.497/0.509/0.515), 60 → 0.437. The budget peaks near 30 on both sides, at roughly -0.04 to -0.06 per halving/doubling.

## Conclusion
Falsified, undertrained side: -0.039 ± 0.013, all of it on the hard folds and untagged/loud tiers. With ps-depth8-e60 (-0.060) it brackets 30 epochs as the depth8 optimum. Held-out val_loss is not a usable proxy for the budget here: it is lower at 15 than at 30 on 7/8 folds and its per-fold minimum sits at epochs 2-19, while sens@fpr0.005 is best at 30. The hard folds' isolated buzzes keep improving after the loss turns. Consequence for the `moderate` run: an epoch count read off pooled val_loss (`train._consensus_epoch`, how the shipped model's budget is picked) would land near ~9 for depth8 and undertrain it; set the budget explicitly (30 on medium; moderate's frame count differs, so steps not epochs may be the thing to hold). Surfaced to Luke, not decided.
