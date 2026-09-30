# ps-nosmooth
## Hypothesis
Label smoothing 0.2 caps targets at 0.9 / floors at 0.1, compressing logits and shrinking the margin at the extreme tails, exactly where the 0.5% FPR threshold is read. Removing it may sharpen the negative tail. Prediction: hard folds with jet/trill false positives improve or stay; headline within noise to slightly up.

Control: `v4-ft-ps` (matched: fine-tuned depth12 trunk + octave-up view, 30 epochs, 0.452). Levers are in the trunk head/loss only; the cached `yamnet_trunk_pitchshift_depth12` embeddings are shared, so nothing is re-extracted.

## Changes
label_smoothing 0.2 -> 0 in the training loss (TRAIN_LS env in train.py; default unchanged).
Config otherwise identical to v4-ft-ps: TRUNK_LR_BACKBONE=1e-5, TRUNK_LR_HEAD=2e-4, TRUNK_BATCH=1024, TRUNK_FP16=1, 30 epochs, BUZZDETECT_CHUNK_FRAMES=48. Env knobs are default-off (`embedders/yamnet_trunk_depth12/embedder.py` copied into the worktree, `03_train/train.py` TRAIN_LS).
Launch env: TRAIN_LS=0
