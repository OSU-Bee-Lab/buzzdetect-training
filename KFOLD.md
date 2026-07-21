# K-fold cross-validation: design

Where we landed on the move from a fixed train/validate/test paradigm to
leave-one-deployment-out cross-validation. Philosophy only — no migration steps.

## The shift

All annotated audio becomes full diel recordings. There is no longer a
train/test/validate distinction in the data — just ~20 days of audio from
different places, annotated across the full diel cycle, assigned to folds at
will. The old `02_set` (event-buffered snips) and separate `04_test` (full
recordings) collapse into one pool of comparable, precision-computable units.

## Fold unit: deployment, not site-date

A fold is one **deployment** — a recorder in one place — with all of its dates
grouped together. The annotation plan records two days per target, so a recorder
recurs across dates. If folds were site-*dates*, day 1 and day 2 of the same
recorder would split across train and test: same location, same colony,
near-duplicate frames. That is the leakage the whole scheme exists to avoid.
Grouping all of a deployment's dates into one fold makes holding it out a
genuinely independent recorder — the literal "train on these hives, test on a
hive I didn't train on."

## No frozen test set

We do not keep a permanent held-out test set. The reasons that normally justify
one don't survive here:

- **No optimism to fix.** In a single CV pass each fold model is tested only on
  its held-out fold, which it never trained on. The CV mean is an honest
  estimate. The shipped model trains on all folds and inherits that estimate
  (mildly conservatively, since it has more data than any fold model). We never
  score the all-data model directly — the CV average stands in for it.
- **A frozen set is one arbitrary draw.** Whatever deployments happened to land
  in it are conditional on where those recorders sat. Leave-one-deployment-out
  makes every deployment the test once and averages over them — a better
  estimate of "generalizes across deployments," which is the actual goal.
- **Data efficiency.** A frozen set bigger than the training pool is a large
  amount of scarce labelled audio permanently sidelined.

The one concern that does survive is **selection bias**: picking winners by the
CV number across many experiments slowly fits the estimate (multiple
comparisons). The fix is not a frozen set — it's discipline. Only promote a
change when it clears fold-to-fold noise, which the CI tooling
(`evaluate_set` / `compare_sets`) already measures. Treat sub-CI deltas as
nothing.

## Early stopping: single-level CV, internal validation split

Each iteration holds out one **pristine test fold** — used only for reporting,
never for any fitting decision. Early stopping monitors a validation split
carved from within the training deployments, not the test fold.

We do **not** hold out a second deployment as validation. Two reasons:

1. **Consistency with the shipped model.** The CV estimate is only meaningful if
   each fold model trains the way the deployed model does. The deployed model
   trains on all deployments and has no held-out fold to stop on, so its stopping
   rule must work without one. If CV stopped on a held-out deployment, the final
   model would either sacrifice a deployment (ship on N−1) or stop differently
   (making the CV number estimate a different procedure). An internal split lets
   every model — each fold and the final — stop identically; only the test fold
   differs (present in CV, absent in the final).
2. **Low stakes.** The classifier is one Dropout + one Dense layer — logistic
   regression on frozen embeddings. Early stopping is a weak regularizer here;
   the exact stopping epoch barely moves the result. Not worth a scarce
   deployment to pin precisely.

Optional polish: draw the internal split at the deployment level (hold one
training deployment as the monitor, rotating) so the early-stop signal comes
from a different recorder than the rest of the training pool. Not a correctness
fix — the correctness requirement is only that the test fold touches no fitting
decision and the final model stops the same way.

## Metrics: sensitivity and FPR are the model metrics

Sensitivity, `P(detection | buzz)`, and false positive rate,
`P(detection | non-buzz)`, are conditioned on the true class, so they're
properties of the classifier — invariant to where the recorder sits. Precision,
`P(buzz | detection)`, mixes in the base rate π (how much buzzing the deployment
contains), which is a property of the deployment, not the model.

Precision is recoverable exactly from the model metrics plus a base rate:

```
precision = sens·π / (sens·π + FPR·(1−π))
```

(Sanity checks: this reproduces the worked examples in the metrics blog post,
and the precision floor equals π — call everything a buzz and
precision → π.) So precision carries no information the model metrics and π
don't already hold. We report sens and FPR as the model metrics and treat
precision as a derived, deployment-conditional quantity.

### Operating point

Targeting "95% precision" was secretly auto-selecting a threshold — it
self-adjusts to the deployment's base rate. Dropping precision means picking the
threshold another way. The base-rate-free replacement is **sensitivity at a
fixed FPR, `f*`** — a point on the ROC curve. To stay continuous with past
reporting, set `f*` to the FPR the current threshold hits at the reference base
rate (π ≈ 0.17). Then "sens @ `f*`" equals the old "sens @ 95% precision" at the
reference base rate, but is now a model property comparable across deployments.

### Aggregation

- **Headline (pooled).** Pool the held-out predictions from all folds into one
  ROC curve, find `f*` once, report sensitivity there. One global threshold —
  which is what actually gets deployed — rather than averaging per-fold
  thresholds, which are meaningless when a fold has few buzzes.
- **Per deployment.** Report sens *and* FPR per deployment, not derived
  precision alone. A carpenter-bee-nest fold and a soybean-field fold differ in
  precision for two unrelated reasons: different base rate (more bees → higher
  precision at the same FPR) and possibly different model skill (different
  acoustic background → different sens/FPR). Precision alone conflates them;
  sens + FPR isolates model skill. Derive per-deployment precision afterward for
  interpretability, since each held-out fold yields its true π.

### Estimation asymmetry

FPR is the tight estimate — there's a flood of non-buzz frames. Sensitivity is
the noisy one, estimated from scarce buzz frames, and noisiest per deployment: a
soybean fold with a handful of buzzes gives a wide sensitivity CI, a
carpenter-bee nest with constant activity gives a tight one. Surface per-
deployment CIs rather than hiding uneven confidence behind a pooled average.

## Summary

- One pool of full-diel recordings; folds assigned at will.
- Fold = deployment (all its dates grouped), leave-one-deployment-out.
- No frozen test set; CV mean is the estimate, selection bias managed by CI
  discipline.
- Pristine test fold per iteration; early stopping on an internal validation
  split; final model stops the same way.
- Report sensitivity and FPR (model metrics); precision derived per deployment
  from π. Headline = pooled sens @ `f*`; also per-deployment sens + FPR with CIs.
