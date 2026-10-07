# ps-depth4
## Hypothesis
If depth6 still gains, extend to layer 3 cut (tail = layers 4-14), the earliest cut before the spatial map grows to 24x16x128 (4x the cache of depth8). Falsifier: gain turns over, or hard folds fall.

Control: `v4-ft-ps` (0.452) and ps-depth10 / ps-depth8 draws; config identical to ps-depth8 (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
New embedders yamnet_trunk_depth4 / yamnet_trunk_pitchshift_depth4, cut at layer3_pointwise_conv_relu.

Run config (batch 24): ps-depth6 OOMed at batch 1024 (train step) and at scoring chunk 1024, so this branch carries ps-depth6's `TRUNK_ACCUM` and `SCORE_CHUNK` commits. Depth 4's activations are larger still: `TRUNK_BATCH=256 TRUNK_ACCUM=4` (batch 1024's update, BN frozen), `SCORE_CHUNK=128`, and `TRUNK_STREAM=1` (the 16 GB two-view pool exceeds what the in-RAM lowmem path holds on the 23 GB host). `run_depth4.sh` holds the exact command; it waits on ps-depth6's pid first.

## Results
Wall time: ~5 h for 8 folds (02:05-07:07), TRUNK_STREAM=1, batch 256 x accum 4.

== vs v4-ft-ps
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.499 | 0.508 | +0.009 | 0.044 | 32 |
| 53 | 0.517 | 0.646 | +0.129 | 0.062 | 28 |
| 1_11 | 0.575 | 0.606 | +0.031 | 0.028 | 26 |
| 1_143 | 0.559 | 0.649 | +0.090 | 0.034 | 22 |
| 1_150 | 0.312 | 0.278 | -0.034 | 0.043 | 21 |
| 1_95 | 0.201 | 0.309 | +0.108 | 0.035 | 46 |
| 1_37 | 0.480 | 0.609 | +0.129 | 0.063 | 14 |
| 1_114 | 0.471 | 0.581 | +0.110 | 0.041 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.452 → this 0.523 (+0.071 ± 0.016)
- inclusive (sensitivity), same thresholds: 0.372 → 0.435 (+0.063)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.843 | 0.801 | -0.042 | 122 |
| untagged | 0.474 | 0.544 | +0.070 | 2418 |
| background | 0.404 | 0.482 | +0.078 | 1874 |
| quiet | 0.110 | 0.143 | +0.033 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.


== vs ps-depth6
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.472 | 0.508 | +0.036 | 0.055 | 32 |
| 53 | 0.604 | 0.646 | +0.042 | 0.027 | 28 |
| 1_11 | 0.551 | 0.606 | +0.055 | 0.030 | 26 |
| 1_143 | 0.631 | 0.649 | +0.018 | 0.037 | 22 |
| 1_150 | 0.278 | 0.278 | +0.000 | 0.034 | 21 |
| 1_95 | 0.256 | 0.309 | +0.053 | 0.038 | 46 |
| 1_37 | 0.645 | 0.609 | -0.036 | 0.038 | 14 |
| 1_114 | 0.384 | 0.581 | +0.197 | 0.037 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.478 → this 0.523 (+0.045 ± 0.013)
- inclusive (sensitivity), same thresholds: 0.397 → 0.435 (+0.038)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.854 | 0.801 | -0.053 | 122 |
| untagged | 0.501 | 0.544 | +0.043 | 2418 |
| background | 0.434 | 0.482 | +0.048 | 1874 |
| quiet | 0.137 | 0.143 | +0.006 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.


== vs ps-depth8-repeat
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.568 | 0.508 | -0.060 | 0.042 | 32 |
| 53 | 0.522 | 0.646 | +0.124 | 0.063 | 28 |
| 1_11 | 0.585 | 0.606 | +0.021 | 0.027 | 26 |
| 1_143 | 0.626 | 0.649 | +0.023 | 0.043 | 22 |
| 1_150 | 0.380 | 0.278 | -0.102 | 0.043 | 21 |
| 1_95 | 0.236 | 0.309 | +0.073 | 0.037 | 46 |
| 1_37 | 0.641 | 0.609 | -0.032 | 0.030 | 14 |
| 1_114 | 0.512 | 0.581 | +0.069 | 0.027 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.509 → this 0.523 (+0.014 ± 0.014)
- inclusive (sensitivity), same thresholds: 0.421 → 0.435 (+0.014)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.819 | 0.801 | -0.018 | 122 |
| untagged | 0.531 | 0.544 | +0.013 | 2418 |
| background | 0.444 | 0.482 | +0.038 | 1874 |
| quiet | 0.126 | 0.143 | +0.017 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.



[exited with code 0]

Interpretation. Headline 0.523 vs control 0.452 (+0.071 ± 0.016), vs depth6 0.478 (+0.045 ± 0.013) and vs depth8-repeat 0.509 (+0.014 ± 0.014; full single-run SD ~0.014-0.016). Against depth8's two-draw mean 0.512 it is +0.011: within noise, below the ~0.027 bar for a repeat. The ladder now reads 0.452 (12) -> 0.480 (10) -> 0.512 (8) -> 0.478 (6) -> 0.523 (4): the 8 and 4 points agree, so depth6's 0.478 looks like the odd draw (or a real dip; one sample each, unsure). The best reading is a plateau around 0.51-0.52 from depth 8 down.

Folds vs depth8-repeat: 53 +0.124 ± 0.063 (unsure), 1_95 +0.073 ± 0.037 (weak positive on the jet fold), 1_114 +0.069 ± 0.027 (positive); 1_150 -0.102 ± 0.043 (a loss on the hardest fold; vs control it is -0.034 ± 0.043, unsure), 1_29 -0.060 ± 0.042 (unsure); others within SD. vs control, every fold but 1_150 gains. Tiers vs depth8-repeat: background +0.038, quiet +0.017, loud -0.018 (122 frames, small).

## Conclusion
Depth 4 (layers 4-14 trainable) ties depth 8 (0.523 vs 0.512 two-draw mean, +0.011 ± ~0.015) at ~2x the compute and 4x the cache. No repeat. The fine-tune depth ladder has saturated at depth 8; depth 8 stays the cheaper default. Next gains need something other than trainable depth (LR schedule, augmentation, epochs). 1_150 is the one fold the deeper cut hurts vs depth8 (-0.10 ± 0.04).
