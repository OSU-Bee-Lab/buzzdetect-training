# batchnorm

## Hypothesis
BatchNormalization of input embeddings before the linear probe stabilizes the embedding
distribution across classes, improving the decision boundary for buzz detection.
YAMNet produces ReLU-activated 1024-dim embeddings with variable per-dimension magnitudes.
BN normalizes each dimension independently across the batch, making the feature distribution
more uniform and potentially easier for the linear probe to separate.

## Changes
- `03_train/train.py`: add `tf.keras.layers.BatchNormalization()` before `Dropout(0.2)`
- No other changes; same defaults (label_smoothing=0.2, lr=0.002, min_delta=0.002,
  patience=50, epochs=400)

## Results
- Baseline (low-delta, no BN): 8 runs, mean=0.224, 95% CI=[0.207, 0.242]
- This experiment (batchnorm v1–v5): 0.156, 0.162, 0.164, 0.173, 0.175  mean=0.166  median=0.164  95% CI=[0.156, 0.176]
CIs don't overlap — clear negative result. BN hurts substantially (~5.8pp mean drop).
Very low variance (std=0.008) compared to no-BN baseline, suggesting BN makes training
more deterministic but worse. Likely cause: BN statistics estimated on training embeddings
don't generalize to the test set's embedding distribution (different recording conditions).

## Conclusion
BatchNormalization on input embeddings hurts substantially and consistently. The BN running
statistics learned from training audio don't transfer to test audio. Not worth pursuing.
