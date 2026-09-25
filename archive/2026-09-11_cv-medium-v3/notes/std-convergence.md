# std-convergence

## Hypothesis

`exp/input-standardization` added a frozen `Normalization` layer (adapt()-ed on
training folds, not `BatchNormalization`) ahead of the probe on
`yamnet_combined`. Headline was flat vs `yamnet-combined` (0.222 -> 0.218), but
on the 8 folds with >=3000 val frames it added +0.014 (6/8 up). 10 of 11 folds
ran the full 400-epoch cap without early stopping ever firing (median ~120
epochs unstandardized), and `best_val_loss` came in higher despite sens
improving in most folds. The fixed Adam LR (0.002) and the
`min_delta=0.002`/`patience=50` stopping rule were tuned against unstandardized
inputs and never revisited once `Normalization` changed the gradient scale the
optimizer sees.

If a lower LR and/or a stopping rule matched to the new gradient scale lets
early stopping actually fire, the resulting `best_val_loss`/sens read is the
honest number for this structure; if convergence still doesn't fire even after
the resweep, the structure itself (not just the hyperparameters) is suspect.

## Changes

Ported the `Normalization` layer from `exp/input-standardization` unmodified
(`03_train/train.py::_train_one`, between `Input` and `Dropout`, `adapt()`-ed
on `data.train_tf` stripped of labels). No other structural change.

Added sweep-only CLI knobs to `03_train/train.py` / `03_train/main.py`, used
only in this experiment:
- `--learning-rate` (threaded into `Adam(learning_rate=...)`, both per-fold and
  shipped-model training)
- `--min-delta` (threaded into `EarlyStopping(min_delta=...)`)
- `--only-folds` — restricts which rotate folds take a *turn as the held-out
  fold* for cheap diagnosis; does not touch `folds.csv` or any fold's role, and
  every other rotate fold still trains into the pool normally. Used only for
  the diagnosis phase below, not for the final CV runs logged.
- `--skip-shipped` — skips the final shipped-model training step, so a
  diagnosis run doesn't pay for a model nobody will read.

Also added `--clipnorm` (threaded into `Adam(clipnorm=...)`), not part of the
original plan — added mid-sweep after the first low-LR diagnostic run produced
a real `NaN` blowup (see Results). This is an optimizer-level safeguard, not a
change to the ported `Normalization` layer or the model structure.

No re-extraction: reused the cached `yamnet_combined` embeddings for `medium`
(symlinked from main, unmodified).

## Results

### Cheap diagnosis (2 folds, `--skip-shipped`)

Picked the two largest-`frames_val` rotate folds, one of which
(`Luke - Diel Drivers/2026-04-08/1_150`) was also the biggest per-fold mover in
`exp/input-standardization`'s own comparison (+0.099 there).

First finding, before any LR change was even judged on its merits: at
`lr=0.0005` (no clipping), the `JamesU - MustardBumbler/1_29` fold's loss went
to `NaN` at epoch 41 and both `loss`/`val_loss`/`accuracy` stayed `NaN` for the
rest of the patience window. I checked why: **52 of the 1545 `yamnet_combined`
input dimensions have ~zero variance across this fold's training pool** (521 of
the dims are YAMNet's own sigmoid class scores, most of which never fire on
field audio), so the ported `Normalization` layer divides those dimensions by
`sqrt(var + 1e-7) ≈ 3e-4` — any nonzero value that shows up there later
(including from a slightly different batch/fold) is amplified ~3000x. This is a
property of the *data* interacting with the already-ported, unmodified
`Normalization` layer, not something this sweep introduced or is meant to fix
(the task is a hyperparameter sweep on a fixed structure) — but it does mean
any LR choice for this structure has to survive it, hence adding `--clipnorm`
as a safeguard rather than treating it as another sweep axis to explore.

| fold | frames_val | config | epochs (best) | best_val_loss | sens@fpr0.005 |
|---|---|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 6908 | lr=2e-3, no clip (`exp/input-standardization`, logged) | 400 (399) | 0.459 | 0.454 |
| JamesU - MustardBumbler/1_29 | 6908 | lr=5e-4, no clip | 66 (16) | 0.721 → NaN | 0.000 |
| JamesU - MustardBumbler/1_29 | 6908 | lr=5e-4, clipnorm=1.0 | 65 (15) | 0.734 | 0.000 |
| JamesU - MustardBumbler/1_29 | 6908 | lr=1e-3, clipnorm=1.0 | 60 (10) | 0.718 | 0.209 |
| JamesU - MustardBumbler/1_29 | 6908 | lr=2e-3, clipnorm=1.0 | 400 (397) | 0.534 | 0.450 |
| Luke - Diel Drivers/2026-04-08/1_150 | 4891 | lr=2e-3, no clip (`exp/input-standardization`, logged) | 400 (400) | 0.457 | 0.168 |
| Luke - Diel Drivers/2026-04-08/1_150 | 4891 | lr=5e-4, clipnorm=1.0 | 400 (386) | 0.627 | 0.085 |
| Luke - Diel Drivers/2026-04-08/1_150 | 4891 | lr=1e-3, clipnorm=1.0 | 95 (45) | 0.666 | 0.043 |
| Luke - Diel Drivers/2026-04-08/1_150 | 4891 | lr=2e-3, clipnorm=1.0 | 400 (395) | 0.536 | 0.069 |
| Luke - Diel Drivers/2026-04-08/1_150 | 4891 | lr=2e-3, no clip, epochs_max=800 (seed-noise check, same config otherwise) | 127 (77) | 0.633 | 0.064 |

Reading this:

- **Lower LR does make early stopping fire** (5e-4 and 1e-3 both stop well
  under the 400 cap on both folds, vs. the original config's 399-400/400), so
  the mechanical hypothesis — LR too high relative to the new gradient scale,
  so real improvement never plateaus inside a fixed 50-epoch window — is
  wrong in the direction it predicted. But every lower-LR config converges to
  a *worse* `best_val_loss` and worse-or-equal `sens@fpr0.005` than the
  original uncapped run on both folds. Full-batch training here (`size_batch`
  is larger than any fold's frame count, so one epoch is one gradient step)
  means a lower LR simply covers less ground in the same epoch budget; the
  original config's "never stops" behaviour looks like real, if slow, ongoing
  improvement being cut off by the 400-epoch cap rather than a spurious
  refusal to recognize a plateau.
- **`clipnorm=1.0` at the original LR (2e-3) does not induce earlier
  stopping** (still 397-400/400 on both folds) and mildly *hurts* both val
  loss and sens on both folds versus the uncapped/unclipped original — it
  constrains legitimate gradient steps this structure needs, not just the
  pathological ones from the near-zero-variance dims.
- **The seed-noise check is the most important line in this table.** Same LR
  (2e-3), same patience/min_delta, same fold, no clipping — only the epoch cap
  was raised (400→800, no effect since it stopped at 127 anyway) — and this
  rerun landed at a real early stop (epoch 127, best 77) with a substantially
  worse val_loss (0.633 vs 0.457) and sens (0.064 vs 0.168) than the exact
  same nominal config's logged run. There is no seed control anywhere in this
  pipeline (per README/LOOP.md), and this single rerun shows that run-to-run
  variance from random init alone is comparable in size to every LR effect
  measured above. That confounds the whole diagnosis: I cannot tell how much
  of the lower-LR configs' worse numbers is the LR and how much is an unlucky
  draw, with only one run per cell.

No config in this sweep showed a reliable win over the original
`exp/input-standardization` config on either metric (convergence *and*
quality), so per LOOP.md's cheap-diagnosis gate, **no full 11-fold CV was run**
— nothing here cleared the bar of "worth an expensive confirmation."

## Conclusion

The sweep does not resolve `exp/input-standardization`'s convergence caveat in
either direction. Lowering the LR does make early stopping fire, but every
config that stopped early landed at a worse `best_val_loss` and worse-or-equal
`sens@fpr0.005` than the original run that never stopped — full-batch Adam at
2e-3 looks like it needs the epochs it's taking, not like it's stuck refusing
to recognize a plateau. Gradient clipping at the original LR doesn't help
convergence and costs a bit of quality. And a same-config rerun on the fold
that moved most under standardization landed on a genuinely different
trajectory (early stop at epoch 127 vs. running to the 400 cap, ~0.17
worse val_loss) purely from uncontrolled random init — noise of that size
makes single-run diagnosis on 1-2 folds unreliable for telling a real LR
effect from a bad draw, which is why this sweep stops at cheap diagnosis
rather than spending an 11-fold CV chasing a signal it can't tell apart from
noise.

Net read: **not adopted, still `exp/input-standardization`'s 0.218
(`sensitivity_mean @ fpr0.005`), still caveated** — but the caveat should now
read as "convergence status genuinely unclear, and not fixable by a small
LR/patience/clipnorm sweep" rather than "needs a hyperparameter resweep",
since that resweep is what this experiment was. Two things worth carrying
forward if `yamnet_combined` + `Normalization` is revisited: (1) the 52
near-zero-variance score dimensions are a real fragility in this structure
independent of LR — a NaN blowup is one bad draw away at any LR low enough to
need many steps to get past the region they create; (2) any future read of
this structure's sensitivity number should budget for a repeated-CV noise
check before trusting small per-fold deltas, given the single seed-noise
sample above was as large as the standardization effect itself.
