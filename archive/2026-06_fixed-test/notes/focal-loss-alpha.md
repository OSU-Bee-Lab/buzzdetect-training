<!-- harvested 2026-09-25 from exp/focal-loss-alpha (9cd465f); pinned at refs/archive/focal-loss-alpha -->

<!-- NOTES.md -->
# focal-loss-alpha

## Hypothesis
Focal loss with alpha=0.75 (vs. 0.25 in focal-loss experiment) improves sensitivity at 95% precision.

The focal-loss experiment showed that alpha=0.25 penalizes positive (buzz) examples too aggressively:
- 95% precision: 0.2499 (slightly worse than baseline 0.2506)
- 90% precision: 0.2937 (better than baseline 0.2763) — mechanism works, direction wrong

Alpha controls the class balance: alpha=0.25 weights positives at 25%, negatives at 75%.
For a recall-focused task like buzz detection, we want to weight the positive class more, not less.
Alpha=0.75 inverts this: weights positives at 75%, negatives at 25% — should push the PR curve
in the high-precision direction.

## Existing evidence
- focal-loss experiment: alpha=0.25 hurt primary metric, improved at 90% precision
- Combined-embedder: all-class validation marginally better
- deeper-head/deeper-std: linear probe outperforms MLP — loss function is key lever
- Lin et al. 2017: alpha is tuned per dataset; 0.25 was for COCO, not bioacoustics

## Config
- Embedder: yamnet, standard set, general translation, linear probe
- Validation: all classes (labels_keep_raw=None)
- Loss: FocalLoss(gamma=2.0, alpha=0.75)
- Everything else: same as focal-loss experiment

<!-- experiments/focal-loss-alpha/notes.md -->
# focal-loss-alpha

## Hypothesis
Focal loss with alpha=0.75 (high positive-class weight) improves sensitivity at 95% precision by
correcting the over-penalization of buzz predictions seen in focal-loss (alpha=0.25).

## Changes
Same as focal-loss experiment but with `FocalLoss(gamma=2.0, alpha=0.75)` instead of alpha=0.25.
- `03_train/train.py`: FocalLoss class, all-class validation, FocalLoss(gamma=2.0, alpha=0.75)
- `03_train/write_model_py.py`: compile=False fix

## Reproduction
No external artifacts. All changes in `03_train/train.py` and `03_train/write_model_py.py`.

## Results

| Model | sens @ 95% prec | sens @ 90% prec | prec @ 80% sens |
|-------|-----------------|-----------------|-----------------|
| test_standard (baseline) | 0.2506 | 0.2763 | 0.2563 |
| exp_fullval_std_v1 (best) | 0.2545 | 0.2755 | 0.2574 |
| exp_focal_loss_v1 (alpha=0.25) | 0.2499 | 0.2937 | 0.2402 |
| exp_focal_alpha75_v1 (this) | 0.2406 | 0.3045 | 0.2667 |

## Interpretation
The alpha parameter creates a systematic tradeoff:
- Higher alpha → better at 90% precision (+2.8pp over baseline), worse at 95% precision (-1.4pp vs baseline)
- Lower alpha → slight improvement at 90% precision, marginal degradation at 95%

The focal loss approach fundamentally shifts the model's operating point toward higher recall / lower precision. This is the wrong direction for the target metric (sensitivity at 95% precision). Focal loss does not appear to be a productive lever for this specific operating point.

**Why it fails at 95% precision**: Focal loss with high alpha upweights buzz predictions aggressively. This increases the buzz recall but also increases buzz false positives — the model predicts buzz more often, lowering precision at any given threshold. At 90% precision the increased recall wins out; at 95% precision the false positives dominate.

## Conclusion
Focal loss (all alpha values) is not productive for sensitivity at 95% precision with YAMNet linear probe.
The loss function changes the operating point of the model rather than improving the PR curve.
Next: try label smoothing (improves calibration without shifting the operating point) or a binary buzz classifier.
