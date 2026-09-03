# restore-on-sens

## Hypothesis

`EarlyStopping` stops the run on `val_loss` and *also restores the val_loss
argmin weights* (`restore_best_weights=True`, `03_train/train.py`). But on this
task `val_sens_fpr0.005` — the number the project is judged on — keeps climbing
well past the `val_loss` minimum, on multiple models:

- `unfreeze_more_1e5` folds 1-4: sens peak epoch 16/60/60/64 vs best_epoch
  5/25/91/35; sens@best -> sens@peak gap +0.035 / +0.036 / +0.015 / +0.047.
- Same pattern noted (and written off as monitor noise) in `trunk-ft-stop-sweep`.

Mechanism this makes sense under: `val_loss` is BCE with `label_smoothing=0.2` —
mean over the val bulk, calibration-sensitive, penalises the growing
overconfidence of a fine-tuning trunk. `sens@fpr0.005` is a rank statistic on
the buzz-vs-hard-negative tail, calibration-invariant, threshold re-derived each
epoch. They diverge late in training exactly as loss-vs-AUC diverge in the
literature. The late-epoch sens gain is real discrimination in the operating
region (buzz vs the top 0.5% of negatives), not overfitting to buzz frames.

**Claim:** decoupling *when to stop* (keep `val_loss` + patience — robust, the
6.6x variance reduction from `min_delta=0.002` is real) from *which epoch to
ship* (the epoch that maximised a *smoothed* sens curve) recovers ~+0.02-0.03
mean sens@fpr0.005, with most folds moving up.

## Changes

- New callback `RestoreBestSens` (`03_train/callbacks.py`): tracks a rolling-mean
  (window 5) `val_sens_fpr0.005` curve, snapshots weights at the smoothed
  argmax, restores them in `on_train_end`. `EarlyStopping` keeps
  `monitor='val_loss'`, `patience`, `min_delta` but drops `restore_best_weights`
  (RestoreBestSens owns the restore now). Listed after SensAtFPR (needs its
  `logs` key) and after EarlyStopping.
- `best_epoch` in `summary.json` becomes the smoothed-sens argmax.
  `val_loss_curve` and `val_sens_fpr0.005_curve` both stored (gap diagnostic +
  `resummarize` + shipped-epoch consensus).
- Shipped model epoch count: `_consensus_epoch` gains a sens-curve mode —
  pool the per-fold `val_sens_fpr0.005_curve`s (extend to common length holding
  last value, running-*max*, min-max normalise, frame-weighted average, earliest
  epoch within `tol` of the consensus *peak*). Falls back to the val_loss
  consensus, then to median best_epoch, for old summaries with no sens curve.

## Smoke test

`tools/smoke_model.py` (compile() untouched, but the restore path changed).

## Results

| fold | baseline sens@fpr0.005 | this exp | delta | buzz frames |
|---|---|---|---|---|

- mean sens@fpr0.005: baseline <val> -> this <val>
- <folds up / down, are the movers buzz-heavy, did the ROC gain concentrate at
  0.005 only (gamed-cross-section signature) or spread left half of the ROC>

## Conclusion
