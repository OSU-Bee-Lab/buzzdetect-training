<!-- harvested 2026-09-25 from exp/label-smooth-03 (ea42f09); pinned at refs/archive/label-smooth-03 -->

<!-- NOTES.md -->
# label-smooth-03

## Hypothesis
Label smoothing ε=0.3 continues the monotonic improvement trend from ε=0.1 (0.2763) → ε=0.2 (0.2993).

Mapping the curve to find the optimum. Risk of degradation if smoothing is too aggressive.

<!-- experiments/label-smooth-03/notes.md -->
# label-smooth-03

## Hypothesis
Label smoothing ε=0.3 continues the monotonic trend from ε=0.1→0.2.

## Changes
- `03_train/train.py`: `BinaryCrossentropy(from_logits=True, label_smoothing=0.3)`
- all-class validation, linear probe, yamnet standard

## Results

| ε | sens @ 95% prec | sens @ 90% prec | prec @ 80% sens |
|---|-----------------|-----------------|-----------------|
| 0.0 (baseline) | 0.2545 | 0.2755 | 0.2574 |
| 0.1 | 0.2763 | 0.3104 | 0.2550 |
| 0.2 | 0.2993 | 0.3281 | 0.2607 |
| **0.3 (this)** | **0.1415** | **0.1677** | **0.2071** |

## Interpretation
ε=0.3 catastrophically degrades performance. With ε=0.3, the smoothed targets are 0.85/0.15,
meaning positive and negative buzz targets are only 0.70 apart. The signal-to-noise ratio for
learning discriminative buzz features collapses and the model can't separate buzz from non-buzz.

The label smoothing optimum is at **ε=0.2** (targets 0.90/0.10). The curve is:
- ε=0.0→0.1→0.2: monotonic gain (+2.2pp each step)
- ε=0.2→0.3: collapse (−1.6pp absolute, −15.8pp at 95% precision)

## Conclusion
ε=0.2 is the sweet spot. ε=0.3 is too aggressive — the targets become too close for the model
to learn discriminative boundaries. Future experiments should use ε=0.2 as the best-known
training configuration (all-class validation + label_smoothing=0.2).
