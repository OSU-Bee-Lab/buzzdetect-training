# ps-bntrain
## Hypothesis
The tail's BatchNorm layers are frozen at AudioSet moving statistics. The fine-tuned weights shift the activation distribution, so the frozen stats may be mismatched; letting BN train (batch 1024) lets the tail re-normalise for this pool. Prediction: modest gain; risk is that batch stats differ between training and per-site inference.

Control: `v4-ft-ps` (matched: fine-tuned depth12 trunk + octave-up view, 30 epochs, 0.452). Levers are in the trunk head/loss only; the cached `yamnet_trunk_pitchshift_depth12` embeddings are shared, so nothing is re-extracted.

## Changes
BatchNormalization layers in the tail set trainable (when the backbone LR > 0).
Config otherwise identical to v4-ft-ps: TRUNK_LR_BACKBONE=1e-5, TRUNK_LR_HEAD=2e-4, TRUNK_BATCH=1024, TRUNK_FP16=1, 30 epochs, BUZZDETECT_CHUNK_FRAMES=48. Env knobs are default-off (`embedders/yamnet_trunk_depth12/embedder.py` copied into the worktree, `03_train/train.py` TRAIN_LS).
Launch env: TRUNK_BN_TRAIN=1

## Results

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.499 | 0.535 | +0.036 | 0.026 | 32 |
| 53 | 0.517 | 0.573 | +0.056 | 0.032 | 28 |
| 1_11 | 0.575 | 0.592 | +0.017 | 0.020 | 26 |
| 1_143 | 0.559 | 0.559 | +0.000 | 0.031 | 22 |
| 1_150 | 0.312 | 0.324 | +0.012 | 0.044 | 21 |
| 1_95 | 0.201 | 0.188 | -0.013 | 0.022 | 46 |
| 1_37 | 0.480 | 0.606 | +0.126 | 0.045 | 14 |
| 1_114 | 0.471 | 0.453 | -0.018 | 0.029 | 28 |
- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.452 → this 0.479 (+0.027 ± 0.011)
- inclusive (sensitivity), same thresholds: 0.372 → 0.397 (+0.025)
- folds that missed fpr 0.005: none
| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.843 | 0.854 | +0.011 | 122 |
| untagged | 0.474 | 0.505 | +0.031 | 2418 |
| background | 0.404 | 0.448 | +0.044 | 1874 |
| quiet | 0.110 | 0.130 | +0.020 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |
± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Headline +0.027 ± 0.011 (eval SD; ~0.014 with training noise, so ~1.9 SD combined: suggestive, not proven). 1_37 +0.126 ± 0.045 clear up (14 events, thin fold); 53 +0.056 ± 0.032, 1_29 +0.036 ± 0.026 up-ish; 1_150 +0.012, 1_95 -0.013, 1_114 -0.018 flat; 1_11 +0.017, 1_143 0. Tiers: background +0.044, untagged +0.031, quiet +0.020, loud +0.011: all leaned up, none large. No fold moved down beyond its SD.

## Conclusion
Letting the tail's BatchNorm train gave the best headline of the pool experiments so far (+0.027), with no fold down beyond noise and the background tier up. Hard folds did not move. Worth a repeat seed / combining with other levers before adopting; also check per-site inference behaviour (BN statistics are trained with batch 1024 but inference uses moving stats).
