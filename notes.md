# ps-depth8-e60
## Hypothesis
The era's training lead (`ps-depth8`, three draws 0.515/0.509/0.497) trains the 30-epoch budget, while the deployed teacher recipe (`v4-ft-ps-e60`, depth12) uses 60: at depth12, 60 epochs gave +0.016 (weak, ~1 SD). Depth8 fine-tunes four more blocks (layers 8-14) at the same LR, so it has more to fit and plausibly more to gain from a longer budget; or it overfits sooner (the depth6 notes raised that, untested). This decides the epoch budget for the final `moderate` run on the depth8 recipe.

Prediction: headline above ps-depth8-r3 (0.497) by more than 0.03, i.e. above the three-draw mean (0.507) by ~0.02, with `untagged` up. Falsified if within ~0.02 of the three-draw mean (the budget doesn't matter at depth8) or below it (60 epochs overfits the deeper tail).

Control: `ps-depth8-r3` (same main recipe, 2026-10-10; annotations unchanged since 2026-09-25); the other draws `ps-depth8` 0.515 and `ps-depth8-repeat` 0.509.

## Changes
None in code. `--epochs 60` instead of 30; otherwise ps-depth8-r3's run (`TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, `--embedder yamnet_trunk_pitchshift_depth8`, medium, translation general).

## Results
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.469 | -0.017 | 0.040 | 32 |
| 53 | 0.550 | 0.526 | -0.024 | 0.030 | 28 |
| 1_11 | 0.573 | 0.564 | -0.009 | 0.030 | 26 |
| 1_143 | 0.586 | 0.560 | -0.026 | 0.041 | 22 |
| 1_150 | 0.435 | 0.350 | -0.085 | 0.051 | 21 |
| 1_95 | 0.284 | 0.215 | -0.069 | 0.026 | 46 |
| 1_37 | 0.648 | 0.593 | -0.055 | 0.038 | 14 |
| 1_114 | 0.414 | 0.220 | -0.194 | 0.069 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.437 (-0.060 ± 0.015)
- inclusive (sensitivity), same thresholds: 0.411 → 0.359 (-0.052)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.780 | -0.076 | 122 |
| untagged | 0.524 | 0.461 | -0.063 | 2418 |
| background | 0.406 | 0.393 | -0.013 | 1874 |
| quiet | 0.130 | 0.122 | -0.008 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

Buzz logit SD 0.66-1.32 on every fold (no collapse). Every fold reports best epoch 60 (= final), as did r3 at 30: the val_loss minimum sits at the end of whatever budget is set (the schedule anneals to it), so "best" says nothing about overfitting. The cross-run comparison does: final val_loss is higher at 60 than at 30 on 7 of 8 folds (1_29 0.687 vs 0.690 the exception; e.g. 53 0.676 vs 0.662, 1_95 0.756 vs 0.748, 1_114 0.753 vs 0.747).

Headline -0.060 ± 0.015 vs ps-depth8-r3 (eval SD only; ~4 SD, past the ~0.014 combined single-run SD by a wide margin), and -0.070 vs the three-draw mean 0.507; below the lowest of five plain depth8 draws (0.464-0.515). Every fold is down. Hard folds lose most: 1_114 -0.194 ± 0.069, 1_95 -0.069 ± 0.026, 1_150 -0.085 ± 0.051 (each beyond its eval SD, though identical runs scatter up to ~0.1 per fold), 1_37 -0.055 ± 0.038 unsure; rich folds within SD. Tiers: loud -0.076 (122 frames), untagged -0.063, background -0.013, quiet -0.008: the loss is in isolated detections.

## Conclusion
Falsified, on the overfit side: 60 epochs hurts depth8 (-0.060 ± 0.015), with losses concentrated in untagged/loud and the hard folds, and higher held-out val_loss on 7/8 folds. Unlike depth12 (+0.016 at 60), the deeper tail fits the training deployments past what transfers. The moderate run on depth8 should keep 30 epochs or fewer; whether <30 helps is untested (`ps-depth8-e15` follows).
