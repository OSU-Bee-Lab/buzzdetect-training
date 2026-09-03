# restore-on-sens

## Hypothesis

`EarlyStopping` stops the run on `val_loss` and *also restores the val_loss
argmin weights* (`restore_best_weights=True`). But `val_sens_fpr0.005` — the
number the project is judged on — keeps climbing well past the `val_loss`
minimum on several models (`unfreeze_more_1e5` folds 1-4: at_best -> peak gap
+0.035 / +0.036 / +0.015 / +0.047; noted and written off as monitor noise in
`trunk-ft-stop-sweep`).

Mechanism: `val_loss` is BCE with `label_smoothing=0.2` — mean over the val
bulk, calibration-sensitive, penalises the growing overconfidence of a
fine-tuning trunk. `sens@fpr0.005` is a rank statistic on the buzz-vs-
hard-negative tail, calibration-invariant, threshold re-derived each epoch.
They diverge late in training the way loss and AUC do.

**Claim:** decouple *when to stop* (keep `val_loss` + patience) from *which
epoch to ship* (argmax of a smoothed sens@fpr curve). Recovers ~+0.02-0.03 mean
sens@fpr0.005, most folds up.

## Changes

- `RestoreBestSens` callback: tracks a trailing 5-epoch rolling mean of
  `val_sens_fpr0.005`, snapshots weights at the *earliest* epoch reaching the
  smoothed max (strict `>`), restores in `on_train_end`. `EarlyStopping` keeps
  `monitor='val_loss'`, `patience`, `min_delta` but `restore_best_weights=False`.
- `summary.json` gains `restored_on`, `loss_argmin_epoch`,
  `val_sens_fpr0.005_curve` (kept alongside `val_loss_curve` for the within-run
  comparison and `resummarize`).
- Shipped-model epoch count (`_consensus_epoch`): new sens-curve mode — pool the
  per-fold smoothed sens curves, running-max, frame-weighted, earliest epoch
  within `stop_tol` of the consensus peak. Falls back to val_loss consensus,
  then median best_epoch.

## Results

Config: `medium` / `yamnet` (frozen 1024-d probe) / `general`, 11 rotating
folds, `main_commit` eb9ecf3.

**Headline: 0.219** sens@fpr0.005 per deployment (shipped epoch count 152).

### Within-run comparison — the clean one (same trained curves, read at three
different restore epochs; no seed or data-drift confound)

| restore rule | mean epoch | mean sens@fpr0.005 |
|---|---|---|
| old: `EarlyStopping` argmin, `min_delta=0.002` | ~75 | **0.2050** |
| true `val_loss` global argmin | ~113 | 0.2134 |
| **new: smoothed-sens argmax** | ~94 | **0.2190** |

new vs old: **mean +0.0140, 9 folds up / 0 down / 2 flat** — direction clean,
no fold harmed.

Decomposition of the +0.014:
- **~+0.008** is just *restoring at the true val_loss minimum instead of the
  last `min_delta=0.002` improvement* — i.e. `stopping-rule-scale` from
  `IDEAS.md`, nothing to do with sensitivity. `trunk-ft-stop-sweep` called
  `min_delta` a "non-lever" but couldn't resolve it under seed noise; read
  within-run off the same curves it's a small real effect on the frozen probe.
- **~+0.006** is the genuine sens-vs-loss divergence (smoothed-sens argmax vs
  val_loss global argmin). That is *inside* the noise floor (`noise-floor-cv`:
  median per-fold |delta| ~0.017, headline ~0.014).

| fold | epoch ES→Sens | s@ES(old) | s@Sens(new) | delta | val frames |
|---|---|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 65→115 | 0.432 | 0.441 | +0.008 | 6984 |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 99→76 | 0.380 | 0.382 | +0.002 | 4712 |
| Lily Adam - willard/2024-08-07/1_11 | 49→93 | 0.174 | 0.191 | +0.016 | 4730 |
| Lily Adam - wooster/2024-07-26/1_143 | 77→127 | 0.237 | 0.266 | +0.029 | 4708 |
| Diel Drivers/2026-04-08/1_150 | 26→76 | 0.041 | 0.082 | +0.041 | 4947 |
| Diel Drivers/2026-05-06/1_95 | 96→141 | 0.030 | 0.030 | +0.000 | 6628 |
| Various/2025-06-23/1_23 | 90→38 | 0.314 | 0.326 | +0.011 | 315 |
| Various/2025-07-03/1_37 | 116→166 | 0.271 | 0.293 | +0.023 | 4715 |
| Various/2025-08-05/31 | 28→33 | 0.157 | 0.160 | +0.002 | 942 |
| Various/2025-08-12/1_114 | 189→185 | 0.197 | 0.202 | +0.005 | 3768 |
| Various/2025-08-27/48 | 43→93 | 0.021 | 0.036 | +0.015 | 1571 |

### vs shipped baseline `models/yamnet_medium_general` (0.199)

+0.020, 8 up / 1 down (−0.006) / 2 flat. Inflated relative to the within-run
+0.014 by run-to-run noise (`noise-floor-cv`: a same-config rerun moved −0.014).
Baseline's `folds_sx.csv` has identical `frames_val` per fold, so this pair is
more comparable than feared — but it is still a different training run.

### The ROC-shape check we wanted (gain concentrated at 0.5% FPR = "gamed
cross-section", or spread across the left ROC = real)

Not done — `SensAtFPR` only logs the 0.005 target, so it needs predictions at
both weight sets. Deferred; the within-run delta is small enough that it isn't
urgent.

## Conclusion

**Caveated positive, adopt-leaning.** Net +0.014 (9 up / 0 down / 2 flat) vs the
shipped `EarlyStopping` restore rule, and no fold is harmed. But the effect
splits: ~+0.008 is the `min_delta=0.002` restore-slack (a separate lever,
`stopping-rule-scale`), and only ~+0.006 is the sens/loss divergence this
experiment is about — which on the *frozen probe* is inside the noise floor.

The large divergence that motivated this (`unfreeze_more_1e5`: sens +0.03-0.05
past the val_loss argmin) is a **backbone-fine-tuning** phenomenon: the frozen
probe's val_loss barely diverges from sens because the label-smoothing
overconfidence penalty only bites once the trunk starts sharpening. Restore-on-
sens is structurally the right rule (it targets the shipped metric directly, it
does no harm, and it degrades gracefully to val_loss for a fold that can't reach
the FPR) — but the frozen-probe CV can't show its value because the gap it
exploits is small here.

**Next:** rerun a trunk-ft config (`trunk_ft_1e5` or `unfreeze_more_1e5`) with
this restore rule, where sens climbs 0.03-0.05 past val_loss — that is where the
change should pay for itself. Also: fold the "restore at true val_loss argmin,
not the min_delta point" finding into `stopping-rule-scale` — it's a free ~+0.008
that this run isolated cleanly.
