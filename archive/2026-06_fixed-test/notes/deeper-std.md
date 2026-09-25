<!-- harvested 2026-09-25 from exp/deeper-std (93fe460); pinned at refs/archive/deeper-std -->

<!-- NOTES.md -->
# deeper-std

## Hypothesis
The existing `exp/deeper-head` experiment added a Dense(128, relu) hidden layer but trained on `lite`, getting 22.14% — worse than the `test_standard` baseline of 25.06%. However, `test_standard` used the much larger `standard` set. This experiment runs the same deeper head on `standard` to give a fair comparison. If the MLP head genuinely helps, more data should let it win.

## Evidence
- YAMNet embeddings (1024-dim) are generic audio representations; a single linear layer may not have enough capacity to isolate buzz-specific features.
- The `exp/deeper-head` result is confounded: lite < standard in data size. Can't conclude deeper head is worse from that comparison alone.
- MLP probing over pretrained audio embeddings typically outperforms linear probing (HEAR benchmark).

## Changes
- `03_train/train.py`: Add Dense(128, relu) before the output layer (same change as exp/deeper-head)
- Same `standard` set and `yamnet` embedder as the current best baseline (test_standard)
- All other hyperparameters unchanged

<!-- experiments/deeper-std/notes.md -->
# deeper-std

## Hypothesis
The existing `exp/deeper-head` experiment added a Dense(128, relu) hidden layer but trained on `lite`, getting 22.14% sensitivity @ 95% precision. The best baseline (`test_standard`) used the larger `standard` set and got 25.06% with a single Dense layer. Those two results aren't a fair comparison. This experiment applies the same deeper head on the `standard` set to isolate the architecture effect.

## Changes
- `03_train/train.py`: Add `Dense(128, activation='relu')` before the output layer
- All other settings identical to `test_standard`: `standard` set, `yamnet` embedder, `general` translation, 300 max epochs, same optimizer and early stopping

## Reproduction
No external artifacts. Run from the worktree:
```bash
conda run -n buzzdetect-train python 03_train/main.py \
  --model exp_deeper_std_v1 --set standard --embedder yamnet \
  --translation general --epochs 300
```
Then stage 4 via spec-loader snippet in LOOP.md.

Note: stage 4 failed on first attempt with `ModuleNotFoundError: No module named 'models.models'` because `models/models.py` is gitignored and absent from the worktree. Manually symlinked it before re-running.

## Results

Metric | test_standard (baseline) | exp_deeper_std_v1 (this experiment)
---|---|---
Sensitivity @ 95% precision | 25.07% | 19.35%
Sensitivity @ 90% precision | 27.63% | 25.17%
Precision @ 80% sensitivity | 25.63% | 26.17%

Training stopped at epoch 51 (best val_loss at epoch 36: 0.2802; patience=30 with min_delta=0.01 fired earlier than expected — probably min_delta caused premature stop).

The deeper head is worse at the high-precision operating point (–5.7 pp at 95% precision) while marginally better at the low-precision end (+0.5 pp at 80% sensitivity). This is a clear negative result for the MLP head.

## Interpretation
YAMNet embeddings appear well-suited for linear probing at high-precision operating points. Adding a hidden layer may:
1. Introduce overfitting with this dataset size
2. Distort calibration in ways that hurt precision-recall tradeoff near the high-precision end
3. Be stopped prematurely by early stopping before the MLP fully converges (only 51 epochs)

The `deeper-head` experiment on `lite` (22.14%) is consistent with this — dataset size isn't the confound; the architecture itself hurts.

## Conclusion
Avoid MLP heads over YAMNet at the high-precision operating point. Linear probing is better here. Future experiments should focus on different embedders, data augmentation, or set composition changes rather than head architecture.
