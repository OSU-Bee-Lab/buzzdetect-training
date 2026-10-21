# gain-aug-d8
## Hypothesis
Deployments differ in recording level (recorder gain, distance from flowers, enclosure), and the hard folds lean quiet: 1_150's buzz is "very quiet, but still legitimate" (Luke, 2026-09-09), and 406 of 1842 buzz annotations are `_quiet`. YAMNet's log-mel front end makes level an additive offset that the fine-tuned depth8 tail can latch onto. Training on a copy of half of every training fold's frames, each at its own random gain uniform in ±10 dB (eval audio untouched), teaches the tail level invariance and shows it quieter buzz than the pool holds. This differs from `gain4` (E3, negative), which applied one fixed 4x gain to train and eval alike and so compressed headroom everywhere. Prediction: headline >= +0.02 over the matched control `ps-depth8-r3` (0.497; three-draw mean 0.507), with `untagged` up, the `quiet` tier up and 1_150 up. Falsifier: within 0.02 of the control and no tier up.

Control: `ps-depth8-r3`. Companions this batch: `eq-aug-d8` (spectral shape), `snr-aug-d8` (additive noise).

## Results
CV on medium, 8 rotating folds, 30 epochs, depth8 recipe with a copy of half of every training fold's snips at a random gain uniform in ±10 dB. Buzz logit SD healthy on every fold (0.55-1.21).

| fold | ps-depth8-r3 | gain-aug-d8 | delta ± SD |
|---|---|---|---|
| 1_29 | 0.486 | 0.534 | +0.048 ± 0.042 |
| 53 | 0.550 | 0.553 | +0.003 ± 0.027 |
| 1_11 (willard) | 0.573 | 0.575 | +0.002 ± 0.027 |
| 1_143 | 0.586 | 0.613 | +0.027 ± 0.056 |
| 1_150 | 0.435 | 0.380 | -0.055 ± 0.037 |
| 1_95 | 0.284 | 0.243 | -0.041 ± 0.029 |
| 1_37 | 0.648 | 0.602 | -0.046 ± 0.061 |
| 1_114 | 0.414 | 0.368 | -0.046 ± 0.030 |

Headline sensitivity_exclquiet 0.484 vs 0.497 (-0.013 ± 0.014; -0.023 vs the depth8 three-draw mean 0.507). Inclusive 0.403 vs 0.411. Tiers: loud +0.007, untagged -0.016, background +0.030, quiet +0.027. No fold missed FPR 0.005.

## Conclusion
Falsified on the headline: -0.013 ± 0.014 is inside the training-noise floor, not the predicted +0.02. The `quiet` (+0.027, 627 frames) and `background` (+0.030) tiers moved the predicted way, weakly: the quiet tier is outside the headline, and this is one draw. The predicted 1_150 gain did not come (-0.055 ± 0.037), and the hard Diel Drivers folds fell again, as under eq-aug-d8 and snr-aug-d8. Of the batch-40 augmentations, gain was the least harmful (eq -0.008, gain -0.013, snr -0.063), and none lifted a hard fold.
