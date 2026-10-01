# ps-depth8
## Hypothesis
Fine-tuning layers 10-14 (ps-depth10) beat layers 12-14 by +0.028 over two draws (0.473, 0.487 vs 0.452), through the background and untagged tiers. If trainable depth is the lever, cutting two blocks earlier (layer 7, tail = layers 8-14) should extend it; the ladder 12 -> 10 -> 8 tells whether the gain saturates. Risk: more parameters to overfit the 41 training deployments; ~1.3x cost per fold again. Layer 7 output is (6,4,512), the same cache width.

Control: `v4-ft-ps` (0.452) and `ps-depth10` / `ps-depth10-repeat` (0.473 / 0.487), identical config (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
New embedders `yamnet_trunk_depth8` and `yamnet_trunk_pitchshift_depth8` (shared tree, copies of the depth10 ones cut at `layer7_pointwise_conv_relu`). Needs a fresh medium extraction.
