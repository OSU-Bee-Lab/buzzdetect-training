# ps-bntrain-repeat
## Hypothesis
`ps-bntrain` (trainable tail BatchNorm) gave 0.479 vs `v4-ft-ps` 0.452 (+0.027, ~1.9 SD with training noise), the best pool-experiment headline of the era but unconfirmed. A repeat draw with identical config should land near +0.027 if real; near +0 if it was training noise. Control: `v4-ft-ps` (0.452) and the first draw `ps-bntrain` (0.479).

## Changes
Only the `TRUNK_BN_TRAIN` env knob (tail BN trainable when backbone LR > 0) copied into the worktree's `embedders/yamnet_trunk_depth12/embedder.py`. Launch: same as v4-ft-ps plus TRUNK_BN_TRAIN=1, 30 epochs.

## Results
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.499 | 0.512 | +0.013 | 0.041 | 32 |
| 53 | 0.517 | 0.545 | +0.028 | 0.031 | 28 |
| 1_11 | 0.575 | 0.549 | -0.026 | 0.022 | 26 |
| 1_143 | 0.559 | 0.595 | +0.036 | 0.035 | 22 |
| 1_150 | 0.312 | 0.352 | +0.040 | 0.047 | 21 |
| 1_95 | 0.201 | 0.188 | -0.013 | 0.024 | 46 |
| 1_37 | 0.480 | 0.605 | +0.125 | 0.043 | 14 |
| 1_114 | 0.471 | 0.407 | -0.064 | 0.035 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.452 → this 0.469 (+0.017 ± 0.013)
- inclusive (sensitivity), same thresholds: 0.372 → 0.383 (+0.011)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.843 | 0.804 | -0.039 | 122 |
| untagged | 0.474 | 0.491 | +0.017 | 2418 |
| background | 0.404 | 0.428 | +0.024 | 1874 |
| quiet | 0.110 | 0.109 | -0.001 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Headline 0.469, +0.017 +/- 0.013 (eval) vs v4-ft-ps; first draw ps-bntrain was 0.479 (+0.027). Two-draw mean 0.474 (+0.022): the gain survives at about half its single-draw size, between 1 and 2 SD. 1_37 +0.125 reproduces (draw 1 +0.126, 14 events, thin fold). 1_114 -0.064 (draw 1 -0.018) is the only fold clearly down, 1_95 -0.013 same as draw 1 (hard folds flat/down). Tiers: untagged +0.017, background +0.024, loud -0.039 (122 frames, small), quiet 0.

## Conclusion
Weak confirmation: trainable tail BN is a probable small gain (~+0.02) carried by 1_37 and the background tier, with 1_114 (trill fold) the cost; hard folds not helped. Not a detection gain on loud.
