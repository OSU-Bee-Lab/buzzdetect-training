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

CV completed 2026-09-04 19:04 (11/11 folds, wrapper needed 11 attempts — the
per-fold CUDA OOM/restart churn the handoff predicted; all self-healing).

Paired against `trunk_ft_1e5` (`tools/compare_folds.py`), sens@fpr0.005:

| fold | trunk_ft_1e5 | this exp | delta | val frames | neg frames |
|---|---|---|---|---|---|
| Luke - Various Opportunistic/2025-08-12/1_114 | 0.316 | 0.283 | -0.033 | 3768 | 17 |
| Luke - Various Opportunistic/2025-06-23/1_23 | 0.400 | 0.383 | -0.017 | 315 | **1** |
| Luke - Various Opportunistic/2025-08-05/31 | 0.185 | 0.171 | -0.014 | 942 | **2** |
| Lily Adam - One Hive/willard/2024-08-07/1_11 | 0.318 | 0.308 | -0.010 | 4730 | 22 |
| Luke - Diel Drivers/2026-05-06/1_95 | 0.048 | 0.048 | 0.000 | 6628 | 30 |
| Lily Adam - One Hive/wooster/2024-07-26/1_143 | 0.337 | 0.358 | +0.021 | 4708 | 22 |
| Luke - Various Opportunistic/2025-07-03/1_37 | 0.293 | 0.326 | +0.033 | 4715 | 22 |
| JamesU - MustardBumbler/1_29 | 0.469 | 0.506 | +0.037 | 6984 | 24 |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 0.432 | 0.475 | +0.043 | 4712 | 18 |
| Luke - Diel Drivers/2026-04-08/1_150 | 0.068 | 0.171 | +0.103 | 4947 | 24 |
| Luke - Various Opportunistic/2025-08-27/48 | 0.020 | 0.244 | **+0.224** | 1571 | **6** |

- mean sens@fpr0.005: `trunk_ft_1e5` 0.262 → this 0.298 (**+0.036**)
- 6 folds up, 4 down, 1 flat.

**The headline is one fold.** `2025-08-27/48` contributes +0.224 of the +0.036
mean on 6 negative frames, and it is exactly the fold where the new rule made
the most extreme choice: it shipped **epoch 2** where `val_loss` argmin was
epoch 58. A 0.5% FPR threshold placed with 6 negatives is not a measurement.
Dropping the three thin folds (1, 2, 6 negatives) leaves **+0.024 over 8 folds,
5 up / 2 down / 1 flat** — above the ~0.017 median per-fold noise floor, but
only just, and with no seed control.

### The within-run comparison (the informative part)

`best_epoch` (shipped, smoothed-sens argmax) vs `loss_argmin_epoch`:

| fold | shipped | val_loss argmin |
|---|---|---|
| JamesU - MustardBumbler/1_29 | 25 | 13 |
| Lily Adam - willard/2024-08-07/1_11 | 38 | 3 |
| Lily Adam - wooster/2024-07-26/1_143 | 109 | 65 |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 96 | 86 |
| Luke - Diel Drivers/2026-04-08/1_150 | 30 | 3 |
| Luke - Diel Drivers/2026-05-06/1_95 | 133 | 96 |
| Luke - Various Opportunistic/2025-06-23/1_23 | 108 | 135 |
| Luke - Various Opportunistic/2025-07-03/1_37 | 96 | 90 |
| Luke - Various Opportunistic/2025-08-05/31 | 10 | 47 |
| Luke - Various Opportunistic/2025-08-12/1_114 | 44 | 63 |
| Luke - Various Opportunistic/2025-08-27/48 | 2 | 58 |

**The two curves diverge on all 11 folds, often by tens of epochs** — the
opposite of the frozen probe, where they stayed locked together and the rule had
nothing to recover. So the mechanism in the hypothesis is confirmed: on a moving
backbone, label-smoothed BCE and rank-based sens@fpr genuinely disagree about
which epoch is best. 8 of 11 folds ship a *later* epoch than the loss argmin,
which is the predicted "sens keeps climbing past the val_loss minimum" shape.

What does not follow is a reliable gain. The rule picks a materially different
model everywhere, yet only 6/11 folds improve, and the mean is carried by the
fold where it made the wildest choice (epoch 2 of 58+). The divergence is real;
the smoothed-sens argmax is a **noisy** estimator of which side of it is better.

## Conclusion

Mechanism confirmed, payoff not. Restoring on smoothed sens@fpr instead of the
`val_loss` argmin selects a genuinely different epoch on every trunk-fine-tuned
fold — validating why `restore-on-sens` measured almost nothing on a frozen
probe — but the held-out result is +0.036 headline / +0.024 excluding the three
folds whose 0.5% FPR threshold rests on 1-6 negative frames, at 6 up / 4 down /
1 flat. That is a hair over the noise floor with no seed control, so the size
should not be trusted and the direction is only weakly supported.

Read as a lever: **worth keeping on the shortlist, not worth adopting on this
evidence.** The clearest next step is not another restore-rule variant but
better validation-fold negatives — three of eleven folds cannot resolve a 0.5%
FPR at all, and they are the ones that decide this comparison. Any rule that
selects an epoch by a per-fold sens@fpr curve is being steered by those same
thin negatives at *training* time too, which is the most likely reason a real
divergence turns into a noisy selection.
