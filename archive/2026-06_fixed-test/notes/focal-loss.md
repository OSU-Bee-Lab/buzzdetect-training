<!-- harvested 2026-09-25 from exp/focal-loss (41e8ae6); pinned at refs/archive/focal-loss -->

<!-- NOTES.md -->
# focal-loss

## Hypothesis
Replace BinaryCrossentropy with focal loss to improve sensitivity at high precision.

Current training uses BinaryCrossentropy with class weights. Easy-to-classify examples (confident non-buzz frames) dominate gradients because BCE treats all examples equally given class membership. Focal loss downweights well-classified examples via `(1 - p_t)^gamma`, forcing the model's gradient budget toward hard, ambiguous cases near the decision boundary — exactly where precision/recall tradeoffs are determined.

## Existing evidence
- Lin et al. 2017 (RetinaNet): focal loss dramatically improved one-stage object detection on imbalanced datasets
- Rare-event audio detection literature consistently shows focal loss gains when classes are severely imbalanced
- YAMNet embeddings are linearly separable (linear probe > MLP from deeper-head/deeper-std experiments), so the loss function is the primary knob left to tune
- All-class validation marginally better than buzz-only (combined-embedder experiment: +0.4pp at 95% precision)

## Config
- Embedder: yamnet (standard, 1024 dims, existing embeddings)
- Set: standard
- Translation: general
- Head: linear probe (unchanged)
- Validation: all classes (from combined-embedder experiment, best current config)
- Loss: FocalLoss(gamma=2.0, alpha=0.25) replacing BinaryCrossentropy
- Everything else: same as baseline (Adam LR=0.002, class weights, patience=30)

<!-- experiments/focal-loss/notes.md -->
# focal-loss

## Hypothesis
Focal loss (gamma=2, alpha=0.25) improves sensitivity at high precision by downweighting easy-to-classify frames, redirecting gradient budget to ambiguous cases near the decision boundary.

## Changes
- `03_train/train.py`: Added `FocalLoss` class using `tf.nn.sigmoid_cross_entropy_with_logits` as base, with per-element `alpha_t * (1 - p_t)^gamma` focal factor. Replaced `BinaryCrossentropy(from_logits=True)` with `FocalLoss(gamma=2.0, alpha=0.25)`.
- `03_train/train.py`: Validation on all classes (`labels_keep_raw=None`) — inherited from combined-embedder experiment.
- `03_train/write_model_py.py`: Fixed generated `model.py` to use `compile=False` on load (required for models with custom loss classes).
- Model: `exp_focal_loss_v1`, standard set, yamnet embedder, 300 epochs max.
- Training stopped at epoch 39 via early stopping (val_loss plateau ~0.0240).

## Reproduction
No external artifacts. All changes are in `03_train/train.py` and `03_train/write_model_py.py`.
Note: the generated `model.py` requires `compile=False` when loading models trained with custom loss classes — this is now fixed in `write_model_py.py`.

## Results
Metric: sensitivity at 95% precision

| Model | sensitivity @ 95% prec | sensitivity @ 90% prec | precision @ 80% sensitivity |
|-------|------------------------|------------------------|------------------------------|
| test_standard (baseline) | 0.2506 | 0.2763 | 0.2563 |
| exp_fullval_std_v1 (best) | 0.2545 | 0.2755 | 0.2574 |
| exp_focal_loss_v1 (this) | 0.2499 | 0.2937 | 0.2402 |

The primary metric (95% precision) is essentially unchanged (-0.7pp from best, -0.1pp from baseline). At 90% precision, focal loss shows a meaningful improvement (+1.7–1.8pp). At 80% sensitivity, focal loss degrades precision (-1.6pp from baseline).

## Interpretation
The focal loss shifts predictions toward more extreme scores. This improves recall at moderate precision thresholds (90%) but doesn't help — and marginally hurts — at the strict 95% precision threshold. The `alpha=0.25` parameter assigns only 25% weight to positive (buzz) examples vs. 75% to negatives. This is equivalent to making the model more conservative about buzz predictions, which hurts sensitivity at the highest precision points.

The `alpha` parameter is the key lever: `alpha=0.25` is the ImageNet-tuned default from Lin et al. and is likely too low for this domain where buzz recall is the priority. A higher alpha (0.5–0.75) would reduce the negative-class bias and likely shift the curve in the right direction.

## Conclusion
Default focal loss (gamma=2, alpha=0.25) does not improve the 95% precision operating point. Alpha=0.25 penalizes buzz predictions too aggressively. Future agents should try focal loss with higher alpha (0.5 or 0.75) — this experiment showed the mechanism works (performance shifts at 90% precision) but the alpha needs tuning toward the buzz-recall direction.
