# ps-featdrop
## Hypothesis
The tail is fine-tuned on 43 deployments and may overfit site-specific channels. Dropping whole channels (p=0.2) of the layer-11 map per view regularises the fine-tuning at the structural level rather than the readout. Prediction: better on unseen deployments, esp. hard folds; could cost rich folds.

Control: `v4-ft-ps` (matched: fine-tuned depth12 trunk + octave-up view, 30 epochs, 0.452). Levers are in the trunk head/loss only; the cached `yamnet_trunk_pitchshift_depth12` embeddings are shared, so nothing is re-extracted.

## Changes
Dropout(0.2, noise_shape=(None,n_ctx,1,1,512)) on the layer-11 map before the shared tail (per-view channel dropout).
Config otherwise identical to v4-ft-ps: TRUNK_LR_BACKBONE=1e-5, TRUNK_LR_HEAD=2e-4, TRUNK_BATCH=1024, TRUNK_FP16=1, 30 epochs, BUZZDETECT_CHUNK_FRAMES=48. Env knobs are default-off (`embedders/yamnet_trunk_depth12/embedder.py` copied into the worktree, `03_train/train.py` TRAIN_LS).
Launch env: TRUNK_SPDROP=0.2

## Results

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.499 | 0.454 | -0.045 | 0.025 | 32 |
| 53 | 0.517 | 0.506 | -0.011 | 0.037 | 28 |
| 1_11 | 0.575 | 0.549 | -0.026 | 0.028 | 26 |
| 1_143 | 0.559 | 0.559 | +0.000 | 0.021 | 22 |
| 1_150 | 0.312 | 0.361 | +0.049 | 0.043 | 21 |
| 1_95 | 0.201 | 0.215 | +0.014 | 0.020 | 46 |
| 1_37 | 0.480 | 0.618 | +0.138 | 0.055 | 14 |
| 1_114 | 0.471 | 0.425 | -0.046 | 0.031 | 28 |
- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.452 → this 0.461 (+0.009 ± 0.012)
- inclusive (sensitivity), same thresholds: 0.372 → 0.378 (+0.006)
- folds that missed fpr 0.005: none
| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.843 | 0.854 | +0.011 | 122 |
| untagged | 0.474 | 0.489 | +0.015 | 2418 |
| background | 0.404 | 0.365 | -0.039 | 1874 |
| quiet | 0.110 | 0.110 | +0.000 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |
± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Headline +0.009 ± 0.012 (eval SD; ~0.014 with training noise): flat. 1_37 +0.138 ± 0.055 is the one clear mover (14 events, thin fold, up); 1_29 -0.045 ± 0.025, 1_114 -0.046 ± 0.031 lean down; 1_150 +0.049 ± 0.043, 1_95 +0.014 ± 0.020, others within SD (unsure). Tiers: background -0.039 (down), untagged +0.015, loud +0.011, quiet 0.

## Conclusion
Per-view channel dropout (p=0.2) on the layer-11 map did not help: headline flat, hard folds 1_150/1_95 at most slightly up within noise, background-tier sensitivity slipped. Not worth a repeat.
