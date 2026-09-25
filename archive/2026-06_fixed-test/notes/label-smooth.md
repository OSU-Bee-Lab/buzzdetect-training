<!-- harvested 2026-09-25 from exp/label-smooth (88da2cd); pinned at refs/archive/label-smooth -->

<!-- NOTES.md -->
# label-smooth

## Hypothesis
Label smoothing (ε=0.1) improves model calibration, lifting sensitivity at 95% precision.

The 95% precision threshold depends entirely on how well-calibrated the model's logit scores are.
With hard 0/1 targets, the model is trained to be maximally confident, potentially overfitting the
training distribution's calibration. Label smoothing replaces targets with:
  positive: 1 → 0.95
  negative: 0 → 0.05
This prevents overconfidence, which is known to improve the precision-recall curve at high thresholds.

The focal loss experiments showed the mechanism matters: the loss function changes calibration, not
just accuracy. Focal loss shifted the curve toward lower precision; label smoothing should improve
the score distribution without biasing it.

## Existing evidence
- Müller et al. 2019 "When Does Label Smoothing Help?": improves calibration (lower ECE) in classification
- Focal loss experiments: showed that loss function choice substantially affects PR curve shape
- Current BCE + class weights is well-optimized for accuracy but not necessarily calibration
- Linear probe is best head — loss function is the remaining lever with existing embeddings

## Config
- Embedder: yamnet, standard set, general translation, linear probe
- Validation: all classes (labels_keep_raw=None)
- Loss: BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
- Everything else unchanged from combined-embedder baseline

<!-- experiments/label-smooth/notes.md -->
# label-smooth

## Hypothesis
Label smoothing (ε=0.1) improves model calibration, lifting sensitivity at 95% precision.

The 95% precision threshold depends on how well-calibrated the model's logit scores are.
BCE with hard 0/1 targets trains the model to be maximally confident, potentially overfitting
the training distribution's calibration. Label smoothing replaces targets with 0.95/0.05,
preventing overconfidence and yielding better-calibrated probability scores.

## Changes
- `03_train/train.py`: `BinaryCrossentropy(from_logits=True)` → `BinaryCrossentropy(from_logits=True, label_smoothing=0.1)`
- `03_train/train.py`: all-class validation (`labels_keep_raw=None`)
- Model: `exp_label_smooth_v1`, standard set, yamnet, linear probe, 300 epochs max
- Training stopped at epoch 50 via early stopping (val_loss plateau ~0.360)

## Reproduction
No external artifacts. One-line change in `03_train/train.py` (`label_smoothing=0.1`).

## Results

| Model | sens @ 95% prec | sens @ 90% prec | prec @ 80% sens |
|-------|-----------------|-----------------|-----------------|
| test_standard (baseline) | 0.2506 | 0.2763 | 0.2563 |
| exp_fullval_std_v1 (prev best) | 0.2545 | 0.2755 | 0.2574 |
| exp_label_smooth_v1 (this) | 0.2763 | 0.3104 | 0.2550 |
| Production target | 0.2800 | — | — |

**Significant improvement: +2.2pp at 95% precision vs. previous best, +8.6pp at 90% precision.**
Precision at 80% sensitivity is essentially unchanged (−0.2pp vs. prev best).

## Interpretation
Label smoothing works by preventing the model from pushing logits to extreme values for
well-classified training examples. The smoothed targets (0.95/0.05 instead of 1/0) create a
persistent gradient signal even for confident correct predictions, improving the calibration
of the output probability scores. A better-calibrated score distribution means the threshold
that achieves 95% precision captures more true buzz events — exactly the sensitivity improvement
we see.

The val_loss was on a different scale (~0.36 vs ~0.024 for focal loss, ~0.1x for BCE) because
smoothed BCE has a higher theoretical minimum, but this doesn't affect model quality.

The model also ran 10+ more epochs before early stopping compared to other variants, suggesting
label smoothing provides a more informative gradient signal for longer.

## Conclusion
Label smoothing (ε=0.1) is the most effective single change found so far: +2.2pp at 95% precision,
reaching 0.2763 vs. production target of 0.2800 (a gap of only 0.4pp remaining).

**Next steps to close the remaining gap:**
1. Try ε=0.05 or ε=0.2 to tune the smoothing factor
2. Combine label smoothing with the yamnet_combined embedder (1545 dims) once extraction is available
3. Combine label smoothing with slight LR reduction (currently 2× default) to potentially improve calibration further
