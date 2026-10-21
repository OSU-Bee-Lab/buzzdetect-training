# Next era: proposed changes

Written 2026-10-21, while batch 41 finishes the current era. Nothing here has
been implemented yet. The era starts when Luke's new training data is ready;
until then the current era runs to the end of batch 41. The agent loop was told
to stop after that batch (`tools/loop_signal.sh stop`).

**Move disks first: see [`MIGRATION.md`](MIGRATION.md).** Before the next era,
bulk data moves off the nearly full root SSD onto the HDD. Re-check
`MIGRATION.md`'s sizes before acting (they were measured 2026-10-20), and follow
its rules: no moves while a chain runs, symlink before rename, and keep
`.local/stream_scratch` on the SSD.

## Order of work

1. Batch 41 (`binary-d8`, `plain-d8`, `sam-d8`, `ps2-seamfree-d8`) finishes and
   is logged into the current era. Don't touch stage scripts while it runs.
2. Disk migration (`MIGRATION.md`).
3. Build the changes below in a worktree, then merge into main.
4. Close the era: `python tools/archive_era.py --slug <slug>` (preview), then
   `--write`. Fill in the archive README's prose (see "Era-close notes" below).
5. New data in place: rebuild the set, extract, then run the anchors.

## 1. Scoring: pick the epoch from the other folds

**Now:** every rotation trains a fixed `--epochs` and is scored at its final
epoch.

**Change:** still train every rotation to a fixed budget, but score fold *f* at
the epoch where the mean of the *other* folds' sensitivity curves peaks
(`honest_epoch.py`'s `xfold-pooled` rule). That becomes the headline. Keep the
final-epoch score as a secondary column.

Why, measured 2026-10-21 on this era's stored per-fold curves (sensitivity
including quiet buzz, the only per-epoch curve on disk). Every number in this
doc reproduces with `tools/epoch_rule_audit.py` (`runs`, `log`, `draws`):

- **Same answer when a change trains like the control.** Over 78 trunk runs,
  cross-fold minus final epoch is +0.002 on average (SD 0.014).
- **Fair to changes that converge at a different speed.** The fixed budget
  scores those long past their peak. pAUC λ0.3 peaked at epoch 5: −0.082 vs
  control at epoch 30, −0.017 at the cross-fold epoch. binary-d8 peaked at epoch
  4: −0.100 → −0.026. ps-depth8-e60: −0.052 → −0.029. No logged verdict flips
  from loss to gain. 58 of 66 entries re-score; the rest are ensembles, gated
  one-fold runs, or name mismatches.
- **The budget becomes a cost knob only.** Too long wastes compute; it no longer
  moves the score. Per-recipe epoch ladders stop being necessary.

What not to do instead:

- **Never stop on `val_loss`.** Stopping each fold at its own `val_loss`
  minimum was worse than the final epoch in 70 of 78 trunk runs (mean −0.032).
  `val_loss` bottoms around epoch 2–8; sensitivity keeps rising to about epoch
  14 and then plateaus. Loss penalises confident logits; sensitivity at a fixed
  FPR only cares about ranking.
- **Never pick a fold's epoch from its own curve.** The curves are smooth
  (epoch-to-epoch jitter 0.005–0.008 per fold on trunks, ≤0.001 on probes), but
  repeat runs drift apart slowly (0.013–0.025 per fold at a given epoch), so a
  fold's own peak mostly picks up that run's luck. Measured with repeat draws
  (pick the epoch on one draw, score it on the others): +0.021 (depth8, 8
  draws), +0.018 (moderate ps-e60, 3), +0.027 (depth12 ps, 3) of inflation,
  against +0.012, +0.002 and −0.003 of real gain. The inflation also varies
  between identical runs (0.020–0.048 over five depth8 draws) and with curve
  shape, so it does not cancel between arms.

Implementation:

- **Persist per-epoch held-out predictions** (or at least per-epoch curves per
  loudness tier, quiet excluded). Today only the inclusive curve is saved, so
  the headline, the tiers and the event-bootstrap SD can't be read at another
  epoch. This is a few MB per run; the callback already scores the fold every
  epoch.
- `folds_sx.csv`: report every column at the cross-fold epoch, plus the
  final-epoch headline and the chosen epoch per fold.
- The selection must run on the headline definition (quiet excluded), not the
  inclusive curve.
- `tools/eval_sampling_sd.py` and `results.py` read the chosen epoch's
  predictions.
- Docs: README "What actually gets trained" and "There is no `validate` role",
  `03_train/CLAUDE.md` (stopping-rule bullets), `docs/training-proposer-facts.md`
  (the epoch rule), `docs/judging-results.md`.

## 2. Shipped model: train to the sensitivity peak, not the loss floor

`train._consensus_epoch` picks the shipped model's epoch count from the pooled
`val_loss` curves, so it inherits the loss problem above. The teacher now in
buzzdetect, `v4-ft-ps-e60-moderate`, was CV'd for 60 epochs and shipped at
**12**; its siblings `-r2`/`-r3` shipped at 14 and 9. On moderate the curve is
flat from about epoch 14, so that cost little (pooled sensitivity 0.459 at e12
vs 0.467–0.469 at e16–21). On the old probes it cost a lot (`yavf_h1024`: 0.185
at its consensus epoch 17 vs 0.313 at 150).

**Change:** ship at the peak of the pooled sensitivity curve (the same rule as
scoring, over all folds). Keep `val_loss` curves as a diagnostic only.

## 3. Epoch budget

From pooled curves averaged over repeat draws:

| recipe | budget now | plateau | peak | after |
|---|---|---|---|---|
| depth8, medium | 30 | e14–30 (0.397–0.401) | e22 | declines: e35 0.371, e60 0.359 |
| ps-e60, moderate | 60 | e14–60 (flat) | e16–21 | flat |
| depth12 ps, medium | 60 | slow rise to e47 | e47 | flat |

Proposed defaults under cross-fold scoring: **40 epochs on medium-sized sets**
(past the depth8 plateau, room for slower changes), **30 on moderate** (half
the compute of 60 for the same result). Re-check on the new data: read the
anchor's pooled curve and confirm the budget runs past its plateau.

## 4. Label smoothing becomes an experiment, off in the baseline

Luke, 2026-10-21: the baseline shouldn't have label smoothing; it should be an
independent test, as dropout already is (`--dropout`, default 0).

**Change:** add `--label-smoothing` (default 0.0) and record it in
`config_model.json`. It is hardcoded at 0.2 in three places today:
`03_train/train.py:514` (the compile that actually runs),
`train_utils.weighted_bce_loss`'s default, and
`embedders/yamnet_trunk/embedder.py:181` (superseded by train.py's compile, but
it should read the same flag so nobody gets surprised).

**Expect it to come back.** Removing it has been measured twice as a loss:
`ps-nosmooth` this era, −0.039 ± 0.015 on the depth12 ps recipe (−0.041 at the
cross-fold epoch; background tier −0.127), and E1's probe tests (+2.2 pp from
0.2, different metric). So the first experiment of the era should be a
smoothing ladder (0 / 0.1 / 0.2) on both the probe anchor and the trunk recipe.
Expect the no-smoothing anchors to sit a few hundredths below this era's
numbers until it does.

## 5. Anchors

LOOP.md's baseline is the frozen YAMNet linear probe. On new data, run:

- **Probe anchor:** frozen YAMNet linear probe, no dropout, no label smoothing.
- **Recipe anchor:** ps-depth8 (`yamnet_trunk_pitchshift_depth8`,
  `TRUNK_LR_BACKBONE=1e-5`), no label smoothing, **2–3 draws** (each ~3 h).
  Five depth8 draws this era scattered with SD ~0.018 (incl. quiet, final
  epoch), so three give a noticeably steadier baseline than one or two.

Then the smoothing ladder (section 4) sets the working recipe. Update LOOP.md's
Baseline section to name the anchors and the new scoring rule.

## Era-close notes (for the archive README)

- **The re-score table.** Every logged training entry at its final epoch vs at
  the cross-fold epoch. Most move < ±0.011; pAUC λ0.3/λ0.1, ps-depth8-e60 and
  binary-d8 move the most, all early-peaking. Put it in the archive so nobody
  re-litigates those from their logged numbers.
- **binary-d8 peaks at epoch 4.** Its epoch-30 headline (−0.116) overstates the
  loss; at the cross-fold epoch it is −0.026, still worse, mostly on 1_95 and
  1_150. One draw. Supports the view that the named confuser classes (trill,
  plane) teach the trainable layers useful distinctions.
- **Shipped-model epoch bug** (section 2), with the teacher's 12 epochs.
- **The batch-40 augmentations lose on their merits.** They ran 1.5–2× the
  optimizer steps at fixed epochs, but read at the step-matched epoch (constant
  learning rate, so the first 20 epochs of a 1.5× run are a step-matched run)
  they still lose: eq +0.005, gain −0.010, pitch −0.035, snr −0.032 (incl.
  quiet, single draws).
- **Since depth8 (2026-10-06) no single-model change has beaten it.** The
  levers with evidence behind them are data (moderate vs medium on one recipe:
  0.574 vs 0.468) and ensembling (+0.049 medium, held on moderate).

## Open for Luke

- **Ensemble as the standard teacher.** Measured: +0.049 on medium, +0.037 over
  members on moderate. Speed is no longer the constraint, so 3× inference may
  be acceptable.
- **Confuser data.** The hard folds look limited by look-alike negatives (the
  jet at 1_95, drones) and missed labels (1_37), not by recording style: every
  invariance method this era hurt them. More annotated confuser negatives
  touches `01_annotate/`.
- **A finer translation** (confusers that set hard folds' thresholds as their
  own classes) is the other direction binary-d8 points to. Needs a CV, so it
  goes in `IDEAS.md` if wanted.
- **1_114's fragility** to anything that touches the views (two-octave view,
  pitch-aug, SNR noise, cross-fold MixStyle) is worth an investigation.
- **The distillation injunction** lifts when `IDEAS.md`'s training queue is
  drained (Luke, 2026-10-21); the `CLAUDE.md` and `LOOP.md` copies still say
  "until Luke lifts this".
