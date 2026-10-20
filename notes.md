# llrd-depth4
## Hypothesis
IDEAS #35. The depth ladder saturates at 8 (ps-depth4 0.523 ties depth8 at ~2x compute) and depth8 overfits past 30 epochs, so with one uniform backbone LR (1e-5) the lower blocks of a deep tail may move too far from their AudioSet features. Layer-wise LR decay (block L at 1e-5 x 0.7^(14-L): layer14 1e-5 down to layer4 2.8e-7) keeps the lower blocks near pretrained while the top adapts. On depth4 (the deepest tail, where the effect should be largest) it should lift the headline >= 0.02 over the depth8 three-draw mean (0.507) and over ps-depth4 (0.523, the matched no-LLRD depth4 run), with gains on `untagged` and the hard folds. Falsifier: depth4+LLRD within ~0.02 of the depth8 three-draw mean, i.e. trainable depth past 8 stays unusable even with LLRD. A hyperparameter on paper; it decides whether depth past 8 is usable, which is structural.
## Changes
`03_train` gains `TRUNK_LLRD=<decay>`: layer-wise LR decay across the trainable trunk tail. Block L trains at `TRUNK_LR_BACKBONE x decay^(14-L)`, so with decay 0.7 layer 14 is at 1e-5 and layer 4 at 2.8e-7; the head keeps its own LR. The run is the ps-depth4 recipe plus `TRUNK_LLRD=0.7` (medium, `yamnet_trunk_pitchshift_depth4`, 30 epochs, TRUNK_STREAM, batch 256x4, fp16). There was no lite smoke because lite has no depth4 embeddings (the first chain died on it at 03:25).

## Results
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.508 | 0.525 | +0.017 | 0.040 | 32 |
| 53 | 0.646 | 0.531 | -0.115 | 0.067 | 28 |
| 1_11 | 0.606 | 0.556 | -0.050 | 0.026 | 26 |
| 1_143 | 0.649 | 0.577 | -0.072 | 0.030 | 22 |
| 1_150 | 0.278 | 0.306 | +0.028 | 0.062 | 21 |
| 1_95 | 0.309 | 0.236 | -0.073 | 0.033 | 46 |
| 1_37 | 0.609 | 0.597 | -0.012 | 0.028 | 14 |
| 1_114 | 0.581 | 0.393 | -0.188 | 0.051 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.523 → this 0.465 (-0.058 ± 0.016)
- inclusive (sensitivity), same thresholds: 0.435 → 0.386 (-0.049)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.801 | 0.854 | +0.053 | 122 |
| untagged | 0.544 | 0.489 | -0.055 | 2418 |
| background | 0.482 | 0.424 | -0.058 | 1874 |
| quiet | 0.143 | 0.123 | -0.020 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Buzz logit SD per fold: 1.22, 1.08, 0.74, 0.73, 0.57, 0.69, 0.76, 0.76. None collapsed. 1_150 at 0.57 is the lowest but still far from ~0.1. Two val_loss minima came at epoch 3 (1_29, 1_114); the rest fell at epochs 7-28, later than the depth8 recipe's 2-12, which fits a backbone that barely moves.

Headline: 0.465 against the matched control ps-depth4 at 0.523 is **-0.058 ± 0.016** (eval sampling only; about ±0.02 with training noise). Against the three-draw depth8 mean of 0.507 it is -0.042. That is a clear loss: the prediction was >= +0.02 over both.
Per fold:
- 1_114: -0.188 ± 0.051, strong evidence of a loss.
- 53: -0.115 ± 0.067, weak evidence of a loss.
- 1_95: -0.073 ± 0.033, moderate evidence of a loss.
- 1_143: -0.072 ± 0.030, moderate evidence of a loss.
- 1_11: -0.050 ± 0.026, moderate evidence of a loss.
- 1_150: +0.028 ± 0.062, unsure.
- 1_29: +0.017 ± 0.040, unsure.
- 1_37: -0.012 ± 0.028, unsure.

The hard folds 1_95 and 1_114 moved down, and 1_150 did not move measurably.
Tiers: `untagged` fell 0.055 and `background` 0.058; `loud` rose 0.053 (122 frames); `quiet` fell slightly (-0.020).

## Conclusion
Falsified. With layer-wise LR decay 0.7 on the depth4 tail, the lower blocks (layers 4-10, at 2.8e-7 to 2.4e-6) stay at their AudioSet features, and that costs -0.058 ± 0.016 against the uniform-LR ps-depth4. It is also -0.042 below the depth8 three-draw mean. The losses fall on 1_114 (-0.188), 1_95, 1_143 and 1_11, and on `untagged` and `background`. So the lower blocks' adaptation was helping, not overfitting, and protecting them with LLRD does not make depth past 8 usable. If anything, a deep tail wants its lower blocks to move. A milder decay (0.9) or LLRD on depth8 is untested, but this run gives no reason to expect a gain from either.
