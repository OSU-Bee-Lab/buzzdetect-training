# ps-depth8-e60
## Hypothesis
The era's training lead (`ps-depth8`, three draws 0.515/0.509/0.497) trains the 30-epoch budget, while the deployed teacher recipe (`v4-ft-ps-e60`, depth12) uses 60: at depth12, 60 epochs gave +0.016 (weak, ~1 SD). Depth8 fine-tunes four more blocks (layers 8-14) at the same LR, so it has more to fit and plausibly more to gain from a longer budget; or it overfits sooner (the depth6 notes raised that, untested). This decides the epoch budget for the final `moderate` run on the depth8 recipe.

Prediction: headline above ps-depth8-r3 (0.497) by more than 0.03, i.e. above the three-draw mean (0.507) by ~0.02, with `untagged` up. Falsified if within ~0.02 of the three-draw mean (the budget doesn't matter at depth8) or below it (60 epochs overfits the deeper tail).

Control: `ps-depth8-r3` (same main recipe, 2026-10-10; annotations unchanged since 2026-09-25); the other draws `ps-depth8` 0.515 and `ps-depth8-repeat` 0.509.

## Changes
None in code. `--epochs 60` instead of 30; otherwise ps-depth8-r3's run (`TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, `--embedder yamnet_trunk_pitchshift_depth8`, medium, translation general).
