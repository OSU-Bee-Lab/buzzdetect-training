# ps-depth10
## Hypothesis
v4-ft-ps fine-tunes layers 12-14 (cut at layer 11). Fine-tuning has been the largest structural lever this era (+0.051 from 13-14 to ... frozen baseline) and the hard folds (1_95, 1_150) fail on isolated events the AudioSet features may not separate. Cutting two blocks earlier (layer 9, tail = layers 10-14, a dose step up in trainable depth) gives the tail more capacity to re-form features for this pool. Prediction: modest gain, possibly on hard folds; risk is overfitting the 41 train deployments. Layers 7-12 share the (6,4,512) shape, so the cache is the same width.

Control: `v4-ft-ps` (0.452), identical config otherwise (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
New embedders `yamnet_trunk_depth10` and `yamnet_trunk_pitchshift_depth10` (shared tree, copies of the depth12 ones cut at `layer9_pointwise_conv_relu`). Needs a fresh medium extraction.

## Results
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.499 | 0.557 | +0.058 | 0.042 | 32 |
| 53 | 0.517 | 0.533 | +0.016 | 0.033 | 28 |
| 1_11 | 0.575 | 0.554 | -0.021 | 0.028 | 26 |
| 1_143 | 0.559 | 0.604 | +0.045 | 0.035 | 22 |
| 1_150 | 0.312 | 0.315 | +0.003 | 0.038 | 21 |
| 1_95 | 0.201 | 0.229 | +0.028 | 0.024 | 46 |
| 1_37 | 0.480 | 0.584 | +0.104 | 0.051 | 14 |
| 1_114 | 0.471 | 0.406 | -0.065 | 0.030 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.452 → this 0.473 (+0.021 ± 0.013)
- inclusive (sensitivity), same thresholds: 0.372 → 0.388 (+0.016)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.843 | 0.861 | +0.018 | 122 |
| untagged | 0.474 | 0.497 | +0.023 | 2418 |
| background | 0.404 | 0.438 | +0.034 | 1874 |
| quiet | 0.110 | 0.120 | +0.010 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Headline +0.021 ± 0.013 (eval SD; ~0.015 with training noise, ~1.4 SD combined): weak evidence for a small gain. 1_37 +0.104 ± 0.051 up (14 events, thin fold); 1_29 +0.058 ± 0.042 and 1_143 +0.045 ± 0.035 up-ish; 1_95 +0.028 ± 0.024, 1_150 +0.003 flat; 1_114 -0.065 ± 0.030 down (~2 SD, the one loss, same fold that suffered in ps-fast/untiled); 1_11 -0.021, 53 +0.016 unsure. Tiers: background +0.034, untagged +0.023, loud +0.018, quiet +0.010, all leaned up, none large.

## Conclusion
Unfreezing layers 10-14 instead of 12-14 gives a weak, headline-level gain (+0.021) that resembles ps-bntrain's (+0.027, +0.017 repeat) and shares its signature: 1_37 up, hard folds 1_95/1_150 flat, 1_114 down. Costs more per fold (~13 min vs ~10). Not a hard-fold lever; a repeat or combination with BN-train is the only reason to revisit.
