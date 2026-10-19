# ps-depth8-e15
## Hypothesis
`ps-depth8-e60` showed depth8 overfits past 30 epochs (-0.060 ± 0.015 vs ps-depth8-r3), and both its and r3's per-fold held-out val_loss curves bottom out at epochs 2-19 (median ~9), at constant LR. If val_loss tracks sensitivity here, a 15-epoch budget sits nearer the minimum than 30 and lifts the headline, with the gain in `untagged`/`loud` and on the hard folds that lost most at 60 (1_114, 1_95, 1_150). It also halves fit time and sets the epoch budget for the final `moderate` run on depth8.

Prediction: headline above ps-depth8-r3 (0.497) and above the three-draw mean (0.507) by ≥0.02. Falsified if within ~0.02 of the three-draw mean (the 15-30 range is flat) or below it (undertrained: val_loss and sens@fpr disagree under 30).

Control: `ps-depth8-r3` (same recipe, 30 epochs, 2026-10-10); other draws `ps-depth8` 0.515, `ps-depth8-repeat` 0.509. Dose ladder with `ps-depth8-e60` (0.437).

## Changes
None in code. `--epochs 15` instead of 30; otherwise ps-depth8-r3's run (`TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, `--embedder yamnet_trunk_pitchshift_depth8`, medium, translation general).
