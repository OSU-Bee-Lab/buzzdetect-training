# ps-untied
## Hypothesis
v4-ft-ps runs the plain and octave-up views through one shared fine-tuned tail (TimeDistributed). The two views have different statistics (the up view is a pitch-shifted spectrum), so one tail must serve both. Giving each view its own tail (both from AudioSet weights, 2x tail parameters) lets each specialise; the linear head still concatenates the two pooled codes. Prediction: modest gain; risk of overfitting with 2x trainable parameters.

Control: `v4-ft-ps` (0.452), identical otherwise (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
`embedders/yamnet_trunk_depth12/embedder.py` (worktree copy): env knob `TRUNK_UNTIED=1` builds one tail per view and concatenates. Launch env: TRUNK_UNTIED=1.

## Results
30 epochs, 8 folds, TRUNK_UNTIED=1.

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.499 | 0.424 | -0.075 | 0.034 | 32 |
| 53 | 0.517 | 0.530 | +0.013 | 0.032 | 28 |
| 1_11 | 0.575 | 0.549 | -0.026 | 0.026 | 26 |
| 1_143 | 0.559 | 0.577 | +0.018 | 0.020 | 22 |
| 1_150 | 0.312 | 0.350 | +0.038 | 0.051 | 21 |
| 1_95 | 0.201 | 0.191 | -0.010 | 0.023 | 46 |
| 1_37 | 0.480 | 0.539 | +0.059 | 0.035 | 14 |
| 1_114 | 0.471 | 0.501 | +0.030 | 0.028 | 28 |

Headline 0.452 -> 0.458 (+0.006 +/- 0.012); inclusive 0.372 -> 0.379. Tiers: loud +0.011, untagged +0.014, quiet +0.017, background -0.047.

Null headline. 1_29 -0.075 (~2 SD) down, 1_37 +0.059 (~1.7 SD) up, 1_114 +0.030 and 1_150 +0.038 within SD (unsure); hard fold 1_95 flat. Background tier down again, the same signature as the other tail perturbations.

## Conclusion
Separate tails per view do not help: null headline, background tier down, hard folds not lifted. The shared tail is not the bottleneck.
