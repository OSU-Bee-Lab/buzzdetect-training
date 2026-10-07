# ps-depth6
## Hypothesis
ps-depth8 (0.515) extended the 12->10->8 ladder without saturating. Cutting two blocks earlier (layer 5, tail = layers 6-14) tests whether it keeps going. Cache is 2x wider per view (12x8x256).

Control: `v4-ft-ps` (0.452) and ps-depth10 / ps-depth8 draws; config identical to ps-depth8 (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
New embedders yamnet_trunk_depth6 / yamnet_trunk_pitchshift_depth6 (shared tree), cut at layer5_pointwise_conv_relu.

Batch 1024 through layers 6-14 OOMs the 4 GB card in fold 1's first train step (that killed the 2026-10-01 attempt with no log left, and the batch-24 relaunch at 22:26). `03_train/train.py` gains `TRUNK_ACCUM` (Keras `gradient_accumulation_steps`): run at `TRUNK_BATCH=512 TRUNK_ACCUM=2`. BN is frozen and Keras averages micro-batches before clipping, so the update is batch 1024's; a toy check gave bit-identical weights and iteration counts (512x2 vs 1024).

Command: `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=512 TRUNK_ACCUM=2 tools/launch_job.sh train.log -- 03_train/main.py --name ps-depth6 --set medium --embedder yamnet_trunk_pitchshift_depth6 --translation general --epochs 30 -y`

## Results
Against the matched control `v4-ft-ps` (`tools/results.py v4-ft-ps ps-depth6`):

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.499 | 0.472 | -0.027 | 0.062 | 32 |
| 53 | 0.517 | 0.604 | +0.087 | 0.054 | 28 |
| 1_11 | 0.575 | 0.551 | -0.024 | 0.029 | 26 |
| 1_143 | 0.559 | 0.631 | +0.072 | 0.038 | 22 |
| 1_150 | 0.312 | 0.278 | -0.034 | 0.049 | 21 |
| 1_95 | 0.201 | 0.256 | +0.055 | 0.025 | 46 |
| 1_37 | 0.480 | 0.645 | +0.165 | 0.073 | 14 |
| 1_114 | 0.471 | 0.384 | -0.087 | 0.037 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.452 → this 0.478 (+0.026 ± 0.017)
- inclusive (sensitivity), same thresholds: 0.372 → 0.397 (+0.025)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.843 | 0.854 | +0.011 | 122 |
| untagged | 0.474 | 0.501 | +0.027 | 2418 |
| background | 0.404 | 0.434 | +0.030 | 1874 |
| quiet | 0.110 | 0.137 | +0.027 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

Against ps-depth8-repeat (0.509; `tools/results.py ps-depth8-repeat ps-depth6`): 0.478, **-0.031 ± 0.015** (inclusive -0.024).
Per fold vs depth8-repeat: 1_29 -0.096 ± 0.057, 53 +0.082 ± 0.054, 1_11 -0.034 ± 0.031, 1_143 +0.005 ± 0.043,
1_150 -0.102 ± 0.044, 1_95 +0.020 ± 0.022, 1_37 +0.004 ± 0.034, 1_114 -0.128 ± 0.035.
Tiers vs depth8-repeat: loud +0.035, untagged -0.030, background -0.010, quiet +0.011.

Interpretation. Against v4-ft-ps the headline gain (+0.026 ± 0.017 eval, ~±0.02 with training noise) is about
depth10's size and well under depth8's (+0.057/+0.063). Against depth8 (two-draw mean 0.512) it is -0.031 to -0.034,
~2 eval SDs, ~1.5 full SDs for one draw: a probable step down, not a certain one. The loss sits on hard folds:
1_114 -0.128 ± 0.035 and 1_150 -0.102 ± 0.044 are both beyond their SDs; 1_29 -0.096 ± 0.057 is unsure.
1_95 +0.020 ± 0.022 is unsure (the jet fold still at 0.256). 53's +0.082 is within ~1.5 SD. Tiers vs depth8 are
mixed: untagged/background down, loud and quiet up slightly (thin tiers).

Depth ladder (headline, v4-ft-ps = depth12): 0.452 -> 0.480 (depth10, two-draw mean) -> 0.512 (depth8, two-draw mean)
-> 0.478 (depth6, one draw).

## Conclusion
The depth ladder probably peaks at 8. Unfreezing layers 6-7 as well (depth6) gives back about half of depth8's gain
(0.478 vs 0.512, -0.031 ± 0.015), and the loss lands on hard folds 1_114 and 1_150, the folds we want to improve.
With the same 30 epochs and backbone LR 1e-5, the extra ~2 blocks of trainable weights likely over-fit the 43
training folds, or need a lower LR / fewer epochs. That is untested. Not repeated: it is not ahead of depth8, so
the repeat rule doesn't apply. ps-depth4 (queued behind this run, same config) will show whether the decline
continues. ps-depth8 stays the depth to build on.
