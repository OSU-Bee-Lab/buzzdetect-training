# ps2-depth8
## Hypothesis
The octave-up view is this era's largest single lever (v4-ft-ps +0.077 over plain fine-tuning; the pitch grid found "up carries the effect", down and method don't matter). Mechanism: a 100-250 Hz buzz fundamental sits where YAMNet's 64 mel bands are coarse and AudioSet features are tuned for speech/music; shifting up an octave moves it into better-resolved bands. If the gain is that mapping and not saturated at one octave, a third view two octaves up (fundamental at 400-1000 Hz; frame resampled 2:1 twice and tiled 4x) adds sensitivity, most on isolated low-SNR buzz (`1_150`, `1_95`; `untagged`/`loud` tiers).

Prediction: headline > ps-depth8-r3 (0.497) by more than 0.03 and above the three-draw ps-depth8 mean (0.507), with `untagged` up. Falsified if within ~0.03 of the three-draw mean or `untagged` doesn't move. Risk: the 4x tile adds seams (removing the tile seam mattered at `1_114` in `ps-untiled`), and the 0.24 s window loses temporal shape.

Control: `ps-depth8-r3` (same recipe, same main, 2026-10-10; annotations unchanged since 2026-09-25), with `ps-depth8` 0.515 and `ps-depth8-repeat` 0.509 as other draws.

## Changes
New embedder `yamnet_trunk_pitchshift2_depth8` (shared tree): `yamnet_trunk_pitchshift_depth8`'s recipe plus a `(Resample('half'), Resample('half'), Tile(4), Fit())` branch, all three through the shared depth8 tail. Run as ps-depth8-r3: `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1`, `--epochs 30`, but `TRUNK_BATCH=512 TRUNK_ACCUM=2 SCORE_CHUNK=512` (batch 1024 OOMs fold 1 with three views; 512x2 is the same update under frozen BN, as depth 6 ran), set medium, translation general.

## Results
Exit 0, 8/8 folds, 30 epochs each (best epoch 30 on every fold), buzz logit SD 0.63-1.19 (no collapse). `tools/results.py ps-depth8-r3 ps2-depth8`:

- headline `sensitivity_exclquiet` 0.497 → **0.480 (-0.017 ± 0.014)**; inclusive 0.411 → 0.395. Against the three-draw ps-depth8 mean (0.507): -0.027.
- per fold (delta ± eval SD): 1_29 +0.077 ± 0.047, 53 -0.010 ± 0.030, 1_11 +0.004 ± 0.030, 1_143 +0.027 ± 0.035, 1_150 -0.055 ± 0.054, 1_95 -0.048 ± 0.026, 1_37 -0.021 ± 0.049, 1_114 -0.110 ± 0.038.
- tiers: loud -0.039 (122 frames), untagged -0.025, background +0.050 (2 folds), quiet 0.000.

## Conclusion
Falsified. The prediction was > +0.03 over the three-draw mean with `untagged` up; the headline is below every ps-depth8 draw and `untagged` fell. The deficit sits on the hard folds the mechanism targeted (1_150 -0.055 ± 0.054, 1_95 -0.048 ± 0.026) and largest at 1_114 (-0.110 ± 0.038), the fold where the tile seam mattered in `ps-untiled`, so the 4x tile / 0.24 s window risk named in the Hypothesis is the likelier story than "the second octave carries no information". The single-octave gain does not extend to two octaves with tiling; a seam-free two-octave view (pad or longer source window instead of tile) would be the only form left worth trying, and it is weak evidence for it. No repeat earned.
