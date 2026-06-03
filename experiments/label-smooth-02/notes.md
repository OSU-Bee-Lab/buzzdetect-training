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
