# trunk-ft-restore-sens

## Hypothesis

`exp/restore-on-sens` changed *which epoch's weights get kept*: stop on
`val_loss` + patience as before, but restore the epoch that maximised a
smoothed (trailing rolling mean, w=5) `val_sens_fpr0.005` curve rather than the
`val_loss` argmin. On the frozen probe it gained **+0.014** (9 up / 0 down /
2 flat, no fold harmed) — but its own decomposition found that most of that was
not the effect it was testing:

| step | sens@fpr0.005 |
|---|---|
| old `EarlyStopping` restore (`min_delta=0.002` slack) | 0.205 |
| true `val_loss` argmin | 0.213 |
| smoothed-sens argmax | 0.219 |

So ~+0.008 was just restoring at the real `val_loss` minimum instead of the last
`min_delta` improvement — a different lever (`stopping-rule-scale`) — and only
~**+0.006** was the sens/loss divergence the rule actually targets. On a frozen
probe that is inside the noise floor (`noise-floor-cv`: ~0.017 median per-fold,
~0.014 headline).

**The divergence that motivated the rule was never a frozen-probe observation.**
It was seen on `unfreeze_more_1e5` — a *fine-tuned trunk*, where sens@fpr kept
climbing 0.03–0.05 past the val_loss minimum. The mechanism is specific to a
moving backbone: `BinaryCrossentropy(label_smoothing=0.2)` is calibration-
sensitive and penalises the growing confidence of a backbone that is still
specialising, while sens@fpr is a rank statistic with the threshold re-derived
each epoch, so it is calibration-invariant. A frozen 1024-d probe barely moves
its calibration, so the two curves stay locked together and there is nothing for
the new rule to recover. Both the `restore-on-sens` log entry and the trail in
`IDEAS.md` say the same thing: the real test is a trunk-ft rerun.

**Prediction:** on `trunk_ft_1e5`'s config the paired gain should be clearly
larger than the frozen probe's ~+0.006 sens/loss component — the kind of move
most folds make in the same direction. If it comes back flat, the rule is
cosmetic on this pipeline and should be logged as a non-lever, alongside
`trunk-ft-stop-sweep`'s `min_delta` finding.

## Changes

Branched from `exp/trunk-ft` (**not** main) so the comparator's code is
identical: main has since gained `shuffle=False` in `model.fit`, a rewritten
shipped-epoch consensus, and the surprisal writer, none of which should sit
inside this comparison.

One change, ported from `exp/restore-on-sens`:

- `03_train/callbacks.py` — taken wholesale from that branch (its `SensAtFPR`
  differs from this base by a comment only). Adds `_trailing_smoothed` and
  `RestoreBestSens`.
- `03_train/train.py` — `EarlyStopping(restore_best_weights=False)` (it still
  owns *stopping*), `RestoreBestSens` appended after it in the callback list so
  its `on_train_end` runs after the stop decision, `best_epoch` now the
  smoothed-sens argmax, and `val_loss_curve` / `val_sens_fpr0.005_curve` /
  `restored_on` / `loss_argmin_epoch` recorded in `summary.json`.

Deliberately **not** ported: that branch's `_consensus_epoch` rewrite. It only
sets the shipped model's fixed epoch count, which no rotating fold and therefore
no cell of `folds_sx.csv` depends on; this base still takes the median
`best_epoch`, which now means the median smoothed-sens argmax. Keeping it out
holds the diff to the one thing being tested.

Everything else matches `trunk_ft_1e5` exactly: `yamnet_trunk` embedder
(layer-12 cache, symlinked from the `trunk-ft` worktree — no re-extraction),
`medium`, `general`, `--batch 1024 --lr-backbone 1e-5 --lr-head 2e-4`,
patience 50, `min_delta` 0.002, epochs cap 400.

## Results

(pending — see HANDOFF_trunk-ft-restore-sens.md)

## Conclusion

(pending)
