# ps-ema
## Hypothesis
Each fold's headline-relevant weights are the final-epoch weights of a noisy 30-epoch fine-tune (training SD ~0.011-0.018 per fold). An exponential moving average of the weights (momentum 0.998, ~500 steps, a few epochs) is a structural noise reducer: it averages the late trajectory without changing the budget or any data. Prediction: small gain from lower variance, most visible on thin folds; no hard-fold lift expected.

Control: `v4-ft-ps` (0.452), identical otherwise (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
`embedders/yamnet_trunk_depth12/embedder.py` (worktree copy): env knob `TRUNK_EMA=momentum` turns on Keras optimizer weight EMA; fit() swaps EMA weights in at the end. Launch env: TRUNK_EMA=0.998.

## Results
30 epochs, 8 folds, TRUNK_EMA=0.998.

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.499 | 0.409 | -0.090 | 0.041 | 32 |
| 53 | 0.517 | 0.508 | -0.009 | 0.050 | 28 |
| 1_11 | 0.575 | 0.544 | -0.031 | 0.019 | 26 |
| 1_143 | 0.559 | 0.577 | +0.018 | 0.021 | 22 |
| 1_150 | 0.312 | 0.361 | +0.049 | 0.046 | 21 |
| 1_95 | 0.201 | 0.194 | -0.007 | 0.020 | 46 |
| 1_37 | 0.480 | 0.553 | +0.073 | 0.034 | 14 |
| 1_114 | 0.471 | 0.403 | -0.068 | 0.033 | 28 |

Headline 0.452 -> 0.444 (-0.008 +/- 0.012); inclusive 0.372 -> 0.362. Tiers: loud -0.032, untagged +0.004, background -0.076, quiet -0.008.

Null headline. 1_29 -0.090 (~2 SD) and 1_114 -0.068 (~2 SD) down; 1_37 +0.073 and 1_150 +0.049 up (1_150 within SD, unsure). Background tier down again, like cutout/noise; untagged flat, loud slightly down. Hard folds 1_95 flat.

## Conclusion
Weight EMA (0.998) does not help: null headline, background tier down. Final-epoch weight noise is not what limits this config.
