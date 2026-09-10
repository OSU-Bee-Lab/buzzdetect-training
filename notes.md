# yamnet-aves-head-fixed

## Hypothesis
`yamnet-aves-head` ran a `--hidden` ladder (h=0/256/1024) on the frozen
`yamnet_aves` concat under plain val_loss early stopping and came back
inconclusive-leaning-positive: shipped null-to-negative (h256 +0.003, h1024
-0.009), but `best_epoch` collapsed monotonically with head width (1_29
100→44→29; willard 89→50→15) and every non-shipped epoch rule
(own-peak, xfold-median, xfold-pooled) put both hidden widths ~+0.02–0.03 above
the linear control. Classic `aves-mlp-head` stopping-rule confound: a wider head
reaches its val_loss argmin sooner and ships undertrained.

IDEAS.md's standing rule: pair any capacity or normalisation change with a fixed
epoch budget or cross-fold epoch rule *from the start*. This experiment does
that. Run the same h=0/256/1024 ladder at `--fixed-epochs 150` — no early stop,
no restore-best, every rotation trains exactly 150 epochs and ships its final
weights, so all three arms are scored at one identical epoch and the capacity
question is not confounded by when val_loss happened to bottom out. Sens curves
are still persisted, so the primary read is `tools/honest_epoch.py` xfold-pooled
(a shared sub-epoch chosen off the other folds) with folds_sx (epoch 150) as the
secondary.

Prediction: if the hidden layer genuinely helps the representation, h256/h1024
beat h0 by ~+0.02–0.03 xfold-pooled with the gain in the resolvable folds
(1_29, willard, 53). If the `yamnet-aves-head` signal was a stopping-rule
artifact only, the ladder is flat here.

## Changes
- `--fixed-epochs N` flag (03_train/main.py, train.py): rotations train exactly
  N epochs, no EarlyStopping / RestoreTrueBest, final weights shipped. SensAtFPR
  curves still persisted. Shipped-model path unchanged.
- `--hidden` cherry-picked from `exp/shared-trunk-head@4a90546` (same as
  `yamnet-aves-head`).
- Embedder `yamnet_aves` (restored + on main as of 4d9623c). Reads the existing
  cache.

## Runs
- `yavf_h0`    — linear control, fixed 150
- `yavf_h256`  — hidden 256, fixed 150
- `yavf_h1024` — hidden 1024, fixed 150

## Results
| rule | h0 | h256 | h1024 |
|---|---|---|---|

## Conclusion
