<!-- harvested 2026-09-25 from exp/label-smooth-02 (2370171); pinned at refs/archive/label-smooth-02 -->

<!-- NOTES.md -->
# label-smooth-02

## Hypothesis
Label smoothing ε=0.2 pushes calibration further than ε=0.1, potentially closing the remaining
0.4pp gap to the production target (0.2800 sensitivity @ 95% precision).

The label-smooth experiment (ε=0.1) achieved 0.2763 vs. target 0.2800 — the mechanism clearly
works. ε=0.2 applies more aggressive smoothing (targets 0.9/0.1 instead of 0.95/0.05), which
may further improve calibration of the buzz score distribution.

Risk: too much smoothing could degrade performance if it prevents the model from learning
strong buzz-specific features. But given we haven't reached a peak, ε=0.2 is worth testing.

## Config
- Same as label-smooth experiment, except label_smoothing=0.2
- Embedder: yamnet, standard set, general translation, linear probe
- Validation: all classes
- Loss: BinaryCrossentropy(from_logits=True, label_smoothing=0.2)

<!-- experiments/label-smooth-02/notes.md -->
# label-smooth-02

## Hypothesis
Label smoothing ε=0.2 pushes calibration further than ε=0.1, closing the remaining 0.4pp gap to target (0.2800).

## Changes
- `03_train/train.py`: `BinaryCrossentropy(from_logits=True, label_smoothing=0.2)`
- all-class validation, linear probe, yamnet standard

## Reproduction
No external artifacts.

## Results

| Model | sens @ 95% prec | sens @ 90% prec | prec @ 80% sens |
|-------|-----------------|-----------------|-----------------|
| test_standard (baseline) | 0.2506 | 0.2763 | 0.2563 |
| exp_label_smooth_v1 (ε=0.1) | 0.2763 | 0.3104 | 0.2550 |
| exp_label_smooth02_v1 (this) | 0.2993 | 0.3281 | 0.2607 |
| Production target | 0.2800 | — | — |

**Exceeds production target by +1.9pp. Best result to date.**

Trend: ε=0.0→0.1→0.2 each adds ~+2.2pp at 95% precision, monotonically. ε=0.3 likely to improve further.

## Conclusion
ε=0.2 exceeds the 0.28 production target. The calibration improvement continues to scale with ε.
Next: ε=0.3 to continue mapping the curve.
