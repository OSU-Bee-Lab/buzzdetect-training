# ps-nosmooth
## Hypothesis
Label smoothing 0.2 caps targets at 0.9 / floors at 0.1, compressing logits and shrinking the margin at the extreme tails, exactly where the 0.5% FPR threshold is read. Removing it may sharpen the negative tail. Prediction: hard folds with jet/trill false positives improve or stay; headline within noise to slightly up.

Control: `v4-ft-ps` (matched: fine-tuned depth12 trunk + octave-up view, 30 epochs, 0.452). Levers are in the trunk head/loss only; the cached `yamnet_trunk_pitchshift_depth12` embeddings are shared, so nothing is re-extracted.

## Changes
label_smoothing 0.2 -> 0 in the training loss (TRAIN_LS env in train.py; default unchanged).
Config otherwise identical to v4-ft-ps: TRUNK_LR_BACKBONE=1e-5, TRUNK_LR_HEAD=2e-4, TRUNK_BATCH=1024, TRUNK_FP16=1, 30 epochs, BUZZDETECT_CHUNK_FRAMES=48. Env knobs are default-off (`embedders/yamnet_trunk_depth12/embedder.py` copied into the worktree, `03_train/train.py` TRAIN_LS).
Launch env: TRAIN_LS=0

## Results

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.499 | 0.333 | -0.166 | 0.062 | 32 |
| 53 | 0.517 | 0.477 | -0.040 | 0.043 | 28 |
| 1_11 | 0.575 | 0.523 | -0.052 | 0.025 | 26 |
| 1_143 | 0.559 | 0.577 | +0.018 | 0.029 | 22 |
| 1_150 | 0.312 | 0.241 | -0.071 | 0.050 | 21 |
| 1_95 | 0.201 | 0.184 | -0.017 | 0.021 | 46 |
| 1_37 | 0.480 | 0.582 | +0.102 | 0.041 | 14 |
| 1_114 | 0.471 | 0.386 | -0.085 | 0.045 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.452 → this 0.413 (-0.039 ± 0.015)
- inclusive (sensitivity), same thresholds: 0.372 → 0.334 (-0.038)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.843 | 0.802 | -0.041 | 122 |
| untagged | 0.474 | 0.445 | -0.029 | 2418 |
| background | 0.404 | 0.277 | -0.127 | 1874 |
| quiet | 0.110 | 0.095 | -0.015 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Headline -0.039 ± 0.015 (eval SD; ~0.02 with training noise): down, ~2 SD combined. 1_29 -0.166 ± 0.062 is the clear loser; 1_114 -0.085 ± 0.045, 1_150 -0.071 ± 0.050, 1_11 -0.052 ± 0.025 lean down; 1_37 +0.102 ± 0.041 up (thin fold, 14 events); 1_143 +0.018, 1_95 -0.017, 53 -0.040 within SD. Tiers: background -0.127 (large), loud -0.041, untagged -0.029, quiet -0.015.

## Conclusion
Removing label smoothing hurt: headline down and the prediction (hard folds hold or improve) failed, with 1_150 and 1_114 leaning down and the background tier losing 0.127. Label smoothing 0.2 stays. Not worth a repeat.
