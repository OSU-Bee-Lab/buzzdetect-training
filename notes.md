# ps-depth8
## Hypothesis
Fine-tuning layers 10-14 (ps-depth10) beat layers 12-14 by +0.028 over two draws (0.473, 0.487 vs 0.452), through the background and untagged tiers. If trainable depth is the lever, cutting two blocks earlier (layer 7, tail = layers 8-14) should extend it; the ladder 12 -> 10 -> 8 tells whether the gain saturates. Risk: more parameters to overfit the 41 training deployments; ~1.3x cost per fold again. Layer 7 output is (6,4,512), the same cache width.

Control: `v4-ft-ps` (0.452) and `ps-depth10` / `ps-depth10-repeat` (0.473 / 0.487), identical config (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
New embedders `yamnet_trunk_depth8` and `yamnet_trunk_pitchshift_depth8` (shared tree, copies of the depth10 ones cut at `layer7_pointwise_conv_relu`). Needs a fresh medium extraction.

## Results
Depth 8 (tail = layers 8-14): 0.515 vs 0.452 control (+0.063 +/- 0.015 eval-sampling SD); vs depth10-repeat (0.487) +0.028 +/- 0.013, vs first depth10 draw (0.473) +0.042. Ladder 12 -> 10 -> 8: 0.452 -> 0.480 (two-draw mean) -> 0.515, no saturation yet.
Tiers vs control: untagged +0.063, background +0.071, quiet +0.037, loud -0.048 (122 frames), faint 0. Same tiers as depth10, larger.
Folds vs control: 1_37 +0.185 (14 events, +/-0.073), 1_150 +0.086, 1_29 +0.081, 1_95 +0.070 (hard folds up), 1_143 +0.045, 53 +0.043, 1_114 +0.005, 1_11 -0.010. No fold clearly down. Training noise (~0.011-0.018 per fold, headline SD ~0.005) is not in the +/-.

## Conclusion
Trainable depth is a real lever: gain grew monotonically from layer 12 to 10 to 8, via background/untagged tiers, at a loud-tier cost (-0.048, small n). Single draw at depth 8, so repeat before trusting the +0.028 over depth 10. Next: depth 6 or earlier (ladder not saturated), mind per-fold cost and overfitting.
