# noquiet-d8
## Hypothesis
Quiet buzz is a true positive but not what a clear buzz sounds like; some of it is just the rasp of passing wings (Luke, 2026-10-21). Training on it may blur the boundary clear buzz would draw. Dropping every training sample whose buzz tier (`train_utils.buzz_tier`, max over its buzz labels) is `quiet` or `faint` — dropping the sample, not ignoring the label, so it never trains as a negative — should sharpen the boundary for audible buzz. Held-out folds are untouched: quiet buzz still scores in the inclusive number and the quiet tier.

Prediction (IDEAS #39): headline >= +0.02 over the matched control `ps-depth8-r3` (0.497) with `untagged` or `loud` up; the quiet tier likely falls. Falsifier: headline within ~0.02 or down.

Control: `ps-depth8-r3` (same ps-depth8 cache, recipe, 30 epochs, batch 1024).
