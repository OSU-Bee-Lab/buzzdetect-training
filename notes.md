# noquiet-d8
## Hypothesis
Quiet buzz is a true positive but not what a clear buzz sounds like; some of it is just the rasp of passing wings (Luke, 2026-10-21). Training on it may blur the boundary clear buzz would draw. Dropping every training sample whose buzz tier (`train_utils.buzz_tier`, max over its buzz labels) is `quiet` or `faint` — dropping the sample, not ignoring the label, so it never trains as a negative — should sharpen the boundary for audible buzz. Held-out folds are untouched: quiet buzz still scores in the inclusive number and the quiet tier.

Prediction (IDEAS #39): headline >= +0.02 over the matched control `ps-depth8-r3` (0.497) with `untagged` or `loud` up; the quiet tier likely falls. Falsifier: headline within ~0.02 or down.

Control: `ps-depth8-r3` (same ps-depth8 cache, recipe, 30 epochs, batch 1024).

## Changes
- `03_train` training-data path drops every training sample whose buzz tier (`train_utils.buzz_tier`, max over its buzz labels) is `quiet` or `faint`; the sample leaves the set, it does not train as a negative. Held-out folds are untouched. 76 of 1191 training samples dropped in the 1_150 rotation.
- Dropping quiet samples also raises `ins_buzz`'s inverse-frequency weight slightly (fewer buzz samples; `build_weights` not pinned).
- Same ps-depth8 cache, recipe, 30 epochs and batch 1024 as the control `ps-depth8-r3`. The shipped model was not trained.

## Results
Buzz logit SD per fold: 1.27, 1.01, 0.76, 0.79, 0.57, 0.85, 0.78, 1.01. 1_150 (0.57) is a bit below the 0.65-1.2 band, as in pitch-aux-d8; no collapse.

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.567 | +0.081 | 0.051 | 32 |
| 53 | 0.550 | 0.542 | -0.008 | 0.031 | 28 |
| 1_11 | 0.573 | 0.580 | +0.007 | 0.029 | 26 |
| 1_143 | 0.586 | 0.586 | +0.000 | 0.034 | 22 |
| 1_150 | 0.435 | 0.333 | -0.102 | 0.043 | 21 |
| 1_95 | 0.284 | 0.181 | -0.103 | 0.036 | 46 |
| 1_37 | 0.648 | 0.615 | -0.033 | 0.038 | 14 |
| 1_114 | 0.414 | 0.280 | -0.134 | 0.029 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.460 (-0.037 ± 0.013)
- inclusive (sensitivity), same thresholds: 0.411 → 0.379 (-0.032)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.799 | -0.057 | 122 |
| untagged | 0.524 | 0.476 | -0.048 | 2418 |
| background | 0.406 | 0.469 | +0.063 | 1874 |
| quiet | 0.130 | 0.119 | -0.011 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only; with training stochasticity, the headline delta's full SD is about 0.014.

The headline fell -0.037 ± 0.013 (full SD ~0.014, so about 2.6 SD): the falsifier holds, and the effect runs the wrong way. The tiers the prediction named fell too: untagged -0.048 (2418 frames), loud -0.057 (122 frames, unsure). Quiet barely moved (-0.011), and background rose +0.063 (1874 frames), the one tier that improved.

Per fold, the hard folds took the largest losses: 1_114 -0.134 ± 0.029, 1_95 -0.103 ± 0.036, 1_150 -0.102 ± 0.043. 1_29 rose +0.081 ± 0.051 (about 1.6 SD, unsure). The rest are within about 1 SD.

## Conclusion
Dropping quiet/faint buzz from training hurts: headline -0.037 ± 0.013, with untagged (-0.048) and the hard folds (1_150, 1_95, 1_114 each about -0.10 to -0.13) down. Quiet buzz is useful training signal for the audible buzz the headline scores, not noise that blurs its boundary. This supports the standing rule that `_quiet` buzz trains as an ordinary positive. Not worth a repeat.
