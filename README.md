# buzzdetect-training

Trains Keras classifiers that detect insect buzzing in passive acoustic
recordings. Raw audio and annotations go in; a linear/shallow probe over a
frozen audio embedder (YAMNet) comes out.

`CLAUDE.md` holds the command reference and directory layout. This file holds
the design decisions behind them — mainly how folds work, since that choice
drives both what gets trained on and what any reported number means.

## Folds

A fold is a **deployment**: one recorder at one site over one period. Folds are
assigned in `01_annotate/` (one per recorder path) and namespaced by source when
sets are built in `02_set/`, so `Even Sample_Luke - Diel Drivers_2026-04-08_1_150`
is one deployment from the Even Sample campaign.

Deployment is the right unit because it's the thing that varies in the field.
Two frames from the same recorder-day share a microphone, a background, a
species assemblage, and a weather pattern; splitting them across train and test
leaks all of that and flatters the model. Holding out a whole deployment asks
the question that actually matters: how will this do on a recorder we haven't
seen?

### Roles

Not every fold plays the same part. `folds.csv` carries a `role` column:

| role | trained on | scored |
|---|---|---|
| `train` | always | never |
| `rotate` | every iteration but its own | on its own iteration |
| `holdout` | never | always |
| `exclude` | never | never |

Two independent questions — is it ever trained on, is it ever scored — give
four cells, so one enum covers the space.

- **`train`** is for data that's useful ballast but isn't a deployment. The
  2025-06-04 original annotations are the case in point: high volume (~46% of
  annotated audio), but a hodgepodge of annotation strategies with, e.g., no
  nighttime coverage. Training on them helps; scoring on them tells you about
  the annotation process rather than about a recorder.
- **`rotate`** is the leave-one-fold-out set — the diel-representative Even
  Sample deployments. Each takes a turn as the held-out fold.
- **`holdout`** is the old fixed-test semantics, kept for when a deployment
  should stay untouched for a headline number. Not currently used.
- **`exclude`** drops a fold from training and scoring without removing it from
  the set. Useful for quarantining a deployment whose annotations are suspect,
  or for ablations, without rebuilding and re-extracting.

Roles are a **training-time policy**, so `02_set/` ignores them and embeds
everything including `exclude`. Flipping a role should never require
re-extraction.

### There is no `validate` role

Each rotation's held-out fold *is* its early-stopping monitor. A separate
`validate` role would cost a second deployment per rotation, and which
deployment drew the short straw would swing the stopping epoch — several folds
here hold only a few dozen seconds of buzz, where `val_loss` is mostly a
measure of ambient reconstruction.

**Never split within a fold to make a validation set.** Snips from one
deployment share a recorder, a site, a background, and a species assemblage, so
a within-fold split puts the same site on both sides. The stopping signal then
measures within-deployment generalization, which is easier than the
cross-deployment thing we care about, and `val_loss` keeps falling after
cross-site performance has started to degrade. The bias has a known direction:
it stops too late and leaves the model more site-overfit than intended.

The cost of monitoring on the held-out fold instead: each fold's reported score
was measured on the fold that chose its stopping epoch, so per-fold numbers and
the pooled ROC are mildly optimistic. Bounded, and small here — the stopping
epoch is a single coarse scalar (patience 50, `min_delta` 0.002), adjacent
epochs are highly correlated, and selection is on multi-class `val_loss` while
the reported metric is buzz sensitivity at fixed FPR. It also mostly cancels
when *comparing* configurations, since every config carries the same bias; it
does not cancel when quoting an absolute number.

Cheap to remove later if it ever matters: keep monitoring as-is but report each
fold at `median(best_epoch)` of the *other* folds, so the epoch that produced
the quoted number never saw the fold it's scored on. The probe is small enough
(~13k params) to snapshot every epoch and re-score without retraining.

Empirically this is close to moot — the loss curves are flat. On
`yamnet_medium_v2`, every held-out fold plateaus and none turn back up, with a
constant train/val gap of ~0.055. Early stopping is barely doing anything, which
is also why the shipped model's fixed epoch count is safe.

## What gets trained

Per rotation: hold out one `rotate` fold, train on every other `rotate` fold
plus all `train` folds, early-stop on the internal split, score the held-out
fold. Fold models are not kept as binaries — only their training and evaluation
artifacts.

The shipped model trains on everything except `holdout`/`exclude`. Nothing is
held out, so there is nothing clean to monitor — it trains for a fixed
`median(best_epoch)` across the rotations. It's the only model saved with a
binary. `holdout` folds never train, so it can be scored on them directly.

## Reading the results

**Report both pooled and per-deployment.** A training run prints both and writes
the pooled ROC to `folds_pooled_metrics.csv` / `folds_pooled_sx.csv`: every
fold's held-out predictions pooled, threshold picked once. That's frame-weighted
by construction, which matches how the shipped model is actually biased. The
unweighted mean across deployments comes from `folds_summary.csv`. The gap
between the two numbers *is* the answer to "should I weight for volume?":
if they agree, fold identity doesn't matter; if they diverge, one high-volume
deployment is carrying the result and the headline needs that caveat attached.

Don't reweight the training set to equalize folds. Fold volumes here are an
artifact of annotation effort, not of nature, so there's no correct weighting to
recover — and the shipped model will be biased toward high-volume folds either
way. Better to know that than to launder it.

**Don't read the fold-to-fold spread as a confidence interval.** Training pools
overlap by ~90% across rotations, so fold models are highly correlated and the
spread understates true uncertainty about a genuinely new deployment.

**Sensitivity and FPR need different amounts of data.** FPR is estimated from
the flood of non-buzz frames and is tight in every fold. Sensitivity is
estimated from buzz frames only, and in the quietest deployments (tens of
seconds of buzz) it's hopeless — ±0.25 or worse. So let every deployment
contribute to pooled FPR, and treat per-deployment sensitivity as meaningful
only where there's buzz to support it. The quiet folds aren't weak folds;
they're the best false-positive probes available, and nighttime false positives
are the known real-world failure mode (see `log.jsonl`, `yamnet-mask`).

Prefer sensitivity at fixed FPR (`metrics_at_fpr`) over sensitivity at fixed
precision when comparing across folds — precision mixes in each deployment's
base rate, FPR doesn't.

## Status

Implemented: roles read from `folds.csv` and checked against what's extracted,
rotation over `rotate` folds with the held-out fold as the early-stopping
monitor, shipped model at `median(best_epoch)`, per-fold held-out scoring,
pooled ROC plus unweighted mean, `holdout` folds scored with the shipped model,
partial reruns that reassemble the summary from disk.

If `folds.csv` has no `role` column, every fold is treated as `rotate` and a
warning is emitted.

Pending: `04_test` and the `summarize_metrics`/`compare_metrics` scripts still
assume a fixed model plus a hand-curated test corpus and haven't been reworked
to aggregate across folds; read `folds_summary.csv`, `folds_pooled_sx.csv`, or
`folds/<fold>/sx.csv` directly until they are.
