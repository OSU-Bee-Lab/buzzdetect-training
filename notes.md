# snr-aug-d8
## Hypothesis
`aug-snr15-trunk` (2026-09-25) was planned as the fine-tuned companion of `aug-snr15-baseline` (SNR-relative noise on the frozen probe: flat, +0.004 ± 0.006) and abandoned unrun at the era cutover. The frozen-probe null was read as "no trainable capacity to use the augmentation" (embed-mixup's argument); the depth8 tail has seven trainable blocks, the most capacity any trunk recipe has had. White noise at 15 dB below each frame's own RMS, on a copy of every training frame, trains the tail to rely less on low-level spectral texture that a noisier recorder or site would mask. Prediction: headline >= +0.02 over the matched control `ps-depth8-r3` (0.497; three-draw mean 0.507), with `untagged`/`quiet` up and hard low-SNR fold 1_150 up. Falsifier: within 0.02 of the control with no tier up, which would close noise augmentation at the trunk as it closed on the probe.

Control: `ps-depth8-r3` (same recipe and cache, no augmentation). Companion: `eq-aug-d8` (same batch, spectral-shape augmentation).

## Results
CV on medium, 8 rotating folds, 30 epochs, depth8 recipe with SNR-15 dB white-noise copies of every training frame. Buzz logit SD healthy on every fold (0.58-1.26); no collapse.

| fold | ps-depth8-r3 | snr-aug-d8 | delta ± SD |
|---|---|---|---|
| 1_29 | 0.486 | 0.479 | -0.007 ± 0.043 |
| 53 | 0.550 | 0.562 | +0.012 ± 0.027 |
| 1_11 (willard) | 0.573 | 0.585 | +0.012 ± 0.023 |
| 1_143 | 0.586 | 0.598 | +0.012 ± 0.037 |
| 1_150 | 0.435 | 0.315 | -0.120 ± 0.045 |
| 1_95 | 0.284 | 0.233 | -0.051 ± 0.031 |
| 1_37 | 0.648 | 0.602 | -0.046 ± 0.057 |
| 1_114 | 0.414 | 0.102 | -0.312 ± 0.050 |

Headline sensitivity_exclquiet 0.434 vs 0.497 (-0.063 ± 0.014; -0.073 vs the depth8 three-draw mean 0.507). Inclusive 0.357 vs 0.411. Tiers: loud +0.005, untagged -0.067, background +0.012, quiet -0.001. No fold missed FPR 0.005.

## Conclusion
Falsified, and harmful: -0.063 is ~4.5x the training-noise floor. The loss sits in `untagged` and concentrates on 1_114 (-0.312 ± 0.050), 1_150 (-0.120 ± 0.045) and 1_95 (-0.051 ± 0.031), all hard folds. Noise copies of every frame seem to teach the tail that broadband hiss-like texture is uninformative, which costs it on deployments whose buzzes are already near the noise floor. eq-aug-d8 (-0.008) also took 1_150 down -0.120. Additive-noise augmentation is closed at the trunk as it was on the probe (aug-snr15-baseline +0.004).
