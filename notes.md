# ps2-depth8
## Hypothesis
The octave-up view is this era's largest single lever (v4-ft-ps +0.077 over plain fine-tuning; the pitch grid found "up carries the effect", down and method don't matter). Mechanism: a 100-250 Hz buzz fundamental sits where YAMNet's 64 mel bands are coarse and AudioSet features are tuned for speech/music; shifting up an octave moves it into better-resolved bands. If the gain is that mapping and not saturated at one octave, a third view two octaves up (fundamental at 400-1000 Hz; frame resampled 2:1 twice and tiled 4x) adds sensitivity, most on isolated low-SNR buzz (`1_150`, `1_95`; `untagged`/`loud` tiers).

Prediction: headline > ps-depth8-r3 (0.497) by more than 0.03 and above the three-draw ps-depth8 mean (0.507), with `untagged` up. Falsified if within ~0.03 of the three-draw mean or `untagged` doesn't move. Risk: the 4x tile adds seams (removing the tile seam mattered at `1_114` in `ps-untiled`), and the 0.24 s window loses temporal shape.

Control: `ps-depth8-r3` (same recipe, same main, 2026-10-10; annotations unchanged since 2026-09-25), with `ps-depth8` 0.515 and `ps-depth8-repeat` 0.509 as other draws.

## Changes
New embedder `yamnet_trunk_pitchshift2_depth8` (shared tree): `yamnet_trunk_pitchshift_depth8`'s recipe plus a `(Resample('half'), Resample('half'), Tile(4), Fit())` branch, all three through the shared depth8 tail. Run as ps-depth8-r3: `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, `--epochs 30`, set medium, translation general.
