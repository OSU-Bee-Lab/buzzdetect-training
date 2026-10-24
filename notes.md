# mixup-d8
## Hypothesis
Mixup is the standard regulariser for audio tagging CNNs (PANNs), and it has never run on a fine-tuned trunk here: E2's `embed-mixup` (-0.007) mixed post-GAP embeddings under a frozen linear probe, where a convex combination of codes is nearly a no-op for a linear readout. Here the trainable depth8 tail sees the layer-7 maps, so mixing two frames' maps (both views with the same lambda) and their targets is manifold mixup at the input of seven trainable conv blocks: it smooths the decision boundary between buzz and its confusers and shows the tail partial-strength buzz on top of other deployments' backgrounds, a feature-space cousin of 1_150's low-SNR positives. Mixing stays inside the training batch (the rotation's training folds only), so no held-out information enters.

lambda ~ Beta(0.4, 0.4) per frame, partner a random permutation of the batch, every training batch; validation and scoring untouched.

Prediction: headline >= +0.02 over `ps-depth8-r3` (0.497; three-draw mean 0.507), untagged or loud up, 1_150 not down. Falsifier: within 0.02, or the augmentation pattern of eq/snr/gain/pitch-aug-d8 (hard folds down).

Control: `ps-depth8-r3` (same cache, 30 epochs, `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`).
