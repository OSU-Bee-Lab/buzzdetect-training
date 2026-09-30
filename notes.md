# ps-gmp
## Hypothesis
Averaging the last feature map over time-frequency washes out short transients (IDEAS prior: hard folds 1_95/1_114/1_150 fail on isolated ~1 s events). Concatenating a global max pool with the global average pool in the tail preserves them. Prediction: hard folds (1_150, 1_95) up, headline up more than the ~0.014 noise, rich folds flat.

Control: `v4-ft-ps` (matched: fine-tuned depth12 trunk + octave-up view, 30 epochs, 0.452). Levers are in the trunk head/loss only; the cached `yamnet_trunk_pitchshift_depth12` embeddings are shared, so nothing is re-extracted.

## Changes
The tail (layers 12-14) now emits [GAP, GMP] (2048-d per view) instead of GAP (1024-d).
Config otherwise identical to v4-ft-ps: TRUNK_LR_BACKBONE=1e-5, TRUNK_LR_HEAD=2e-4, TRUNK_BATCH=1024, TRUNK_FP16=1, 30 epochs, BUZZDETECT_CHUNK_FRAMES=48. Env knobs are default-off (`embedders/yamnet_trunk_depth12/embedder.py` copied into the worktree, `03_train/train.py` TRAIN_LS).
Launch env: TRUNK_GMP=1
