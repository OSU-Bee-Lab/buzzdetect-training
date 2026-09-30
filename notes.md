# ps-gmp
## Hypothesis
Averaging the last feature map over time-frequency washes out short transients (IDEAS prior: hard folds 1_95/1_114/1_150 fail on isolated ~1 s events). Concatenating a global max pool with the global average pool in the tail preserves them. Prediction: hard folds (1_150, 1_95) up, headline up more than the ~0.014 noise, rich folds flat.

Control: `v4-ft-ps` (matched: fine-tuned depth12 trunk + octave-up view, 30 epochs, 0.452). Levers are in the trunk head/loss only; the cached `yamnet_trunk_pitchshift_depth12` embeddings are shared, so nothing is re-extracted.

## Changes
The tail (layers 12-14) now emits [GAP, GMP] (2048-d per view) instead of GAP (1024-d).
Config otherwise identical to v4-ft-ps: TRUNK_LR_BACKBONE=1e-5, TRUNK_LR_HEAD=2e-4, TRUNK_BATCH=1024, TRUNK_FP16=1, 30 epochs, BUZZDETECT_CHUNK_FRAMES=48. Env knobs are default-off (`embedders/yamnet_trunk_depth12/embedder.py` copied into the worktree, `03_train/train.py` TRAIN_LS).
Launch env: TRUNK_GMP=1

## Results

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.499 | 0.516 | +0.017 | 0.030 | 32 |
| 53 | 0.517 | 0.499 | -0.018 | 0.030 | 28 |
| 1_11 | 0.575 | 0.546 | -0.029 | 0.024 | 26 |
| 1_143 | 0.559 | 0.586 | +0.027 | 0.026 | 22 |
| 1_150 | 0.312 | 0.278 | -0.034 | 0.040 | 21 |
| 1_95 | 0.201 | 0.163 | -0.038 | 0.023 | 46 |
| 1_37 | 0.480 | 0.629 | +0.149 | 0.039 | 14 |
| 1_114 | 0.471 | 0.409 | -0.062 | 0.038 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.452 → this 0.453 (+0.001 ± 0.011)
- inclusive (sensitivity), same thresholds: 0.372 → 0.375 (+0.003)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.843 | 0.854 | +0.011 | 122 |
| untagged | 0.474 | 0.481 | +0.007 | 2418 |
| background | 0.404 | 0.395 | -0.009 | 1874 |
| quiet | 0.110 | 0.123 | +0.013 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Headline +0.001 ± 0.011 (eval SD; ~0.014 with training noise): flat. Per fold: 1_37 +0.149 ± 0.039 is the one clear mover (14 events, thin fold, up); 1_114 -0.062 ± 0.038 (~1.6 SD, unsure/down); 1_143 +0.027, 1_29 +0.017, 53 -0.018, 1_11 -0.029, 1_150 -0.034 ± 0.040, 1_95 -0.038 ± 0.023 all within their SDs (unsure). The hard folds 1_95/1_150/1_114 did not improve; if anything they slipped. Tiers: loud +0.011, untagged +0.007, background -0.009, quiet +0.013; none moved.

## Conclusion
Adding a global max pool alongside the average pool in the fine-tuned tail did not help: headline flat, and the prediction (hard folds up) failed; 1_95/1_150/1_114 all leaned down. The single big gain is on 1_37 (chicory, threshold set by `ambient_background`), which trades against 1_114. Not a detection gain on any tier. Not worth a repeat.
