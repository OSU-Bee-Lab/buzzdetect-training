# ps-bntrain
## Hypothesis
The tail's BatchNorm layers are frozen at AudioSet moving statistics. The fine-tuned weights shift the activation distribution, so the frozen stats may be mismatched; letting BN train (batch 1024) lets the tail re-normalise for this pool. Prediction: modest gain; risk is that batch stats differ between training and per-site inference.

Control: `v4-ft-ps` (matched: fine-tuned depth12 trunk + octave-up view, 30 epochs, 0.452). Levers are in the trunk head/loss only; the cached `yamnet_trunk_pitchshift_depth12` embeddings are shared, so nothing is re-extracted.

## Changes
BatchNormalization layers in the tail set trainable (when the backbone LR > 0).
Config otherwise identical to v4-ft-ps: TRUNK_LR_BACKBONE=1e-5, TRUNK_LR_HEAD=2e-4, TRUNK_BATCH=1024, TRUNK_FP16=1, 30 epochs, BUZZDETECT_CHUNK_FRAMES=48. Env knobs are default-off (`embedders/yamnet_trunk_depth12/embedder.py` copied into the worktree, `03_train/train.py` TRAIN_LS).
Launch env: TRUNK_BN_TRAIN=1
