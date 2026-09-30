# ps-featdrop
## Hypothesis
The tail is fine-tuned on 43 deployments and may overfit site-specific channels. Dropping whole channels (p=0.2) of the layer-11 map per view regularises the fine-tuning at the structural level rather than the readout. Prediction: better on unseen deployments, esp. hard folds; could cost rich folds.

Control: `v4-ft-ps` (matched: fine-tuned depth12 trunk + octave-up view, 30 epochs, 0.452). Levers are in the trunk head/loss only; the cached `yamnet_trunk_pitchshift_depth12` embeddings are shared, so nothing is re-extracted.

## Changes
Dropout(0.2, noise_shape=(None,n_ctx,1,1,512)) on the layer-11 map before the shared tail (per-view channel dropout).
Config otherwise identical to v4-ft-ps: TRUNK_LR_BACKBONE=1e-5, TRUNK_LR_HEAD=2e-4, TRUNK_BATCH=1024, TRUNK_FP16=1, 30 epochs, BUZZDETECT_CHUNK_FRAMES=48. Env knobs are default-off (`embedders/yamnet_trunk_depth12/embedder.py` copied into the worktree, `03_train/train.py` TRAIN_LS).
Launch env: TRUNK_SPDROP=0.2
