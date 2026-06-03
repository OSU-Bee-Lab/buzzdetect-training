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
