# context-pooling

## Hypothesis

`yamnet_context` (the honest context embedder from `exp/context-embedder`, +0.022
over cv-baseline) feeds a flat 3072-d vector — 3 stacked 1024-d frames — into
the usual linear probe, giving every frame's every dimension equal weight. A
learned attention pool (`AttentionPool`: shared `Dense(1)` score per frame,
softmax over the frame axis, weighted sum) should do better than that flat
concatenation by letting the model down-weight less-informative neighbour
frames, at the cost of one small extra layer before the usual
`Dropout(0.2)` + `Dense(n_classes)` head. `POOL_MODE = 'attention'` in
`03_train/train.py`; `'max'` (GlobalMaxPooling1D) exists as a cheap
non-learned control but was not run this pass.

## Changes

`03_train/train.py`: `_build_model` now branches on `_uses_context_pooling`
(true when `embedder.n_embeddings` is a multiple of 1024 greater than 1024).
For a context embedder it reshapes the flat input to `(n_frames, 1024)` and
routes it through `AttentionPool` before the same head every other model uses.
Registered with `@tf.keras.utils.register_keras_serializable` so it survives
the `save(include_optimizer=True) -> load(compile=False)` round trip. No
change to the embedder, extraction, loss, or LR — same `yamnet_context`
embeddings (read-only from main's cache), same Adam 0.002, same
`BinaryCrossentropy(label_smoothing=0.2)`, same `EarlyStopping(patience=50,
min_delta=0.002, restore_best_weights=True)`.

Trained as `context_pooling_attention`, `medium` set, `general` translation,
11 rotating folds + shipped model. Full CV, no retraining needed for this
writeup.

## Results

Paired against both comparators (`compare_folds.py` for cv-baseline; manual
join for context-embedder, whose `folds_sx.csv` predates the `sensitivity_mean`
column rename):

| fold | baseline sens@fpr0.005 | context-embedder sens@fpr0.005 | this exp | delta vs baseline | delta vs context-embedder | frames_val | NaN'd? |
|---|---|---|---|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 0.448 | 0.550 | 0.006 | -0.442 | -0.544 | 6984 | no |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 0.386 | 0.390 | 0.000 | -0.386 | -0.390 | 4712 | no |
| Luke - Various Opportunistic/2025-06-23/1_23 | 0.326 | 0.314 | 0.000 | -0.326 | -0.314 | 315 | yes (epoch 1) |
| Lily Adam - One Hive/wooster/2024-07-26/1_143 | 0.260 | 0.247 | 0.005 | -0.255 | -0.242 | 4708 | yes (epoch 4) |
| Luke - Various Opportunistic/2025-08-05/31 | 0.157 | 0.297 | 0.161 | 0.004 | -0.136 | 942 | yes (epoch 3) |
| Luke - Various Opportunistic/2025-08-12/1_114 | 0.163 | 0.122 | 0.000 | -0.163 | -0.122 | 3768 | no |
| Lily Adam - One Hive/willard/2024-08-07/1_11 | 0.177 | 0.118 | 0.010 | -0.167 | -0.108 | 4730 | yes (epoch 3) |
| Luke - Diel Drivers/2026-04-08/1_150 | 0.028 | 0.068 | 0.000 | -0.028 | -0.068 | 4947 | yes (epoch 2) |
| Luke - Diel Drivers/2026-05-06/1_95 | 0.027 | 0.042 | 0.002 | -0.025 | -0.040 | 6628 | yes (epoch 2) |
| Luke - Various Opportunistic/2025-08-27/48 | 0.013 | 0.023 | 0.020 | 0.007 | -0.003 | 1571 | no |
| Luke - Various Opportunistic/2025-07-03/1_37 | 0.279 | 0.336 | 0.339 | 0.060 | 0.003 | 4715 | no |

- sensitivity_mean @ fpr0.005: cv-baseline **0.206** → this **0.049** (-0.157);
  context-embedder **0.228** → this **0.049** (-0.179)
- Vs cv-baseline: 3 up, 8 down. Vs context-embedder (the fairer comparator,
  since pooling only modifies that model): 1 up, 10 down.

### Early stopping / NaN diagnosis

`folds_summary.csv` shows `best_epoch` 1-4 on 6 of the 11 folds, against
60-130 for a healthy run. Loading `history.pickle` for those folds (had to
`import train` first so the registered `AttentionPool` class deserializes)
shows this is **not** an early-stopping-min_delta artifact — it's outright
numerical collapse:

```
willard/1_11   val_loss: 0.645, 0.595, 0.552, nan, nan, ...   (NaN at epoch 3)
1_150          val_loss: 0.652, 0.621, nan, nan, ...          (NaN at epoch 2)
1_23           val_loss: 0.611, nan, nan, ...                 (NaN at epoch 1)
```

All 6 low-best-epoch folds go to `nan` in both `val_loss` and training `loss`
within 1-4 epochs and never recover; `restore_best_weights` then hands back
the last pre-NaN checkpoint, which is essentially an undertrained/near-random
model — hence sens@fpr0.005 of 0.000-0.02 on those folds. The other 5 folds
(JamesU/1_29, Fit+Fast/53, 1_114, 48, 1_37) trained cleanly for 53-132 epochs,
no NaN.

But those 5 clean folds don't rescue the idea either: their mean sens@fpr0.005
is 0.073 (0.006, 0.000, 0.000, 0.020, 0.339), still well below both
comparators, dragged up almost entirely by one outlier (1_37, +0.06 over
baseline, the fold that already had unusually high sensitivity in both prior
models). Excluding NaN folds entirely still leaves the pooling head weaker
than the flat linear probe on the context embedding, not just a stopped-early
artifact on the numbers as measured.

Likely mechanism for the NaN collapse: `Dense(1)` scoring 3 frames is a small,
poorly-conditioned linear map whose output feeds a `softmax` — with Adam at
0.002 (tuned for the flat linear probe, not for a layer stacked in front of
it) it's plausible for the score logits to blow up in a couple of steps,
saturating the softmax to a one-hot weighting and sending gradients through
that hot frame to infinity/NaN. This wasn't checked against a lower LR or
gradient clipping since LOOP.md's hyperparameter-tuning guidance says not to
chase this without a structural reason, and the clean-fold evidence above
already argues the direction is negative independent of the crash.

`frames_val` is healthy (>3000) on 8 of 11 folds, including most of the big
NaN-fold losses (JamesU, wooster, 1_114, willard, 1_150, 1_95) — those deltas
are trustworthy in direction. The two lowest-frames_val folds (1_23 at 315,
48 at 1571) are the ones where a single-fold read is least reliable, and
they happen to be a down and a trivial up respectively — doesn't change the
picture.

## Conclusion

Attention pooling over `yamnet_context`'s stacked frames, as implemented here
(`Dense(1)` score → softmax → weighted sum, same Adam 0.002 / BCE
label_smoothing=0.2 / patience=50 as every other run), is a large negative:
-0.157 against cv-baseline, -0.179 against the fairer context-embedder
comparator, with 10/11 folds down against the latter. Two things are true at
once and both need to be in the record: (1) 6/11 folds suffered outright NaN
loss collapse within 1-4 epochs — a genuine numerical-stability bug in this
pooling formulation at this LR, not merely `min_delta` tripping early — so
`best_epoch` 1-4 does not mean "converged fast," it means "crashed and kept
the wreckage"; (2) the 5 folds that trained without crashing still
underperformed both comparators (mean sens 0.073 vs 0.206/0.228), so fixing
the instability would not obviously turn this into a win. `trust: caveated` —
the sign is credible under either reading (crashed or clean), but the
magnitude is inflated by the NaN folds and shouldn't be read as "attention
pooling costs -0.157" without first trying a lower LR or `POOL_MODE='max'`
to see whether a stable version narrows the gap. Not worth another full CV
on this exact formulation without that quick stability fix first; leaving it
as a caveated negative rather than reopening it as an IDEA.
