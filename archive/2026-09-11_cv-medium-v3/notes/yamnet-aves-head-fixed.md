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

Fixed 150 epochs, no early stopping, no restore-best. All arms scored at one
identical epoch. Mean sens@fpr0.005 over the 5 rotating folds:

| fold    | yavf_h0 | yavf_h256 | yavf_h1024 | yavf_h1024_r2 |
|---------|---------|-----------|------------|---------------|
| 1_29    | 0.451   | 0.439     | 0.464      | 0.450         |
| 53      | 0.448   | 0.465     | 0.491      | 0.476         |
| willard | 0.259   | 0.269     | 0.294      | 0.287         |
| 1_150   | 0.160   | 0.195     | 0.236      | 0.215         |
| 1_95    | 0.064   | 0.049     | 0.078      | 0.071         |
| **mean**| **0.276** | **0.283** | **0.313**  | **0.300**     |

honest_epoch (no retraining), mean over folds:

| rule         | h0    | h256  | h1024 | h1024_r2 |
|--------------|-------|-------|-------|----------|
| shipped(e150)| 0.276 | 0.284 | 0.313 | 0.300 |
| own-peak     | 0.277 | 0.288 | 0.319 | 0.305 |
| xfold-median | 0.275 | 0.278 | 0.301 | 0.290 |
| xfold-pooled | 0.275 | 0.281 | 0.312 | 0.303 |

**h256: null.** +0.006 to +0.008 over the linear control at every rule, folds
split 3up/2down. A 1792->256 bottleneck buys nothing.

**h1024: +0.030 over the matched linear control, reproducible.** Two draws:
+0.037 (r1, 5 folds up / 0 down) and +0.024 (r2, 4 up / 1_29 flat at -0.001);
r1 vs r2 is -0.013 with all 5 folds uniformly lower on r2 -- seedless training
noise, not a fold pattern. Two-draw mean h1024 0.307 vs h0 0.276.

Per fold, h1024 two-draw mean vs h0: **1_150 +0.066, 53 +0.036, willard +0.032**,
1_95 +0.011, 1_29 +0.006. The gain concentrates in the hard resolvable folds
(1_150, willard) and the second rich fold (53); 1_29 flat both draws.

**The stopping rule was undertraining the whole pipeline.** yavf_h0 -- a plain
linear probe on yamnet_aves, differing from the early-stopped yav_h0 (0.236,
yamnet-aves-head) ONLY in fixed-150 vs val_loss early stop -- is **0.276**,
+0.040 for the linear control alone. own-peak 0.277 == shipped 0.276, so h0 has
genuinely plateaued by epoch 150; this is not overfitting past a peak. Same
~+0.03-0.04 that aves-readout / aves-mlp-head flagged for YAMNet incidentally,
here at full size.

Decomposition of yavf_h1024 (two-draw mean 0.307) vs cv_baseline (0.218), +0.089,
roughly additive:
- yamnet_aves concat vs YAMNet:        ~+0.018  (yav_h0 0.236 vs cv_baseline)
- fixed-150 epochs vs val_loss stop:   ~+0.040  (yav_h0 0.236 -> yavf_h0 0.276)
- 1024 hidden layer vs linear:         ~+0.030  (yavf_h0 0.276 -> yavf_h1024 0.307)

## Conclusion
A 1024-wide ReLU hidden layer on the frozen yamnet_aves concat is +0.030 over
the matched linear control under a fixed epoch budget, direction reproducible
across two draws (4/5 folds up both times, hard folds moving most), where the
same ladder under val_loss early stopping (yamnet-aves-head) read null-to-
negative. That confirms yamnet-aves-head's suspicion: the head helps the
representation and early stopping was hiding it. h256 is null -- the width
matters, a narrow bottleneck does not.

The larger finding is incidental: fixed-150 vs val_loss early stopping is +0.040
for the linear control by itself. The pipeline's stopping rule is leaving ~0.04
on the table before any structural change -- this is xfold-epoch's conclusion,
re-measured, and it now has a second independent motivation (it is load-bearing
for whether a non-linear head reads as positive or negative).

trust: clean on the direction (two draws, 4/5 folds, hard folds); the absolute
+0.030 rests on n=2 with fixed-150 untuned and 1_150 carrying ~0.066 of it at
its known ~0.055 run-to-run noise, so treat the size as approximate.

LEADS:
1. Compose with yamnet_context. context (+0.058 xfold) attacks temporal
   contrast, aves attacks frequency coverage, the hidden layer is a third
   orthogonal lever. yamnet_context_aves + --hidden 1024 --fixed-epochs is the
   obvious stack, though it is now three things at once.
2. Width between 256 and 1024, and > 1024 -- h256 null, h1024 +0.030 is a coarse
   ladder.
3. The fixed-epoch effect deserves its own clean pass (or the xfold-epoch
   cutover Luke deferred). Every structural result in this era may be understated
   by ~0.04.
