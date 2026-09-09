# mech-margin

## Hypothesis

`cv_baseline`'s threshold-setting false positives, pooled across the 5 rotating
folds, are led by **`mech_auto` (75 frames), ahead of `ambient_background` (46)
and `ins_trill` (43)** (IDEAS.md `trill-vs-buzz`, answered 2026-09-08). The
concentration is extreme in one fold: at `Diel Drivers/2026-05-06/1_95`
(sens 0.037) **32 of the 35 frames above its own fpr0.005 threshold are
`mech_auto`**, and it is the only fold whose threshold is positive (+0.173 vs
~-1.7 everywhere else). IDEAS.md names a buzz-vs-`mech_auto` margin as "the
remaining candidate" for that fold twice, after `harmonic-comb` moved it +0.002
in each of two runs (the only fold byte-stable across two nondeterministic runs
— its failure is structural, not stochastic).

The loss is plain per-class weighted BCE: every negative frame costs the same,
so vehicle drone that lands just above the operating point is worth no more to
push down than silence that is already 3 logits below it. The metric is
sensitivity at the 99.5th percentile of negatives, i.e. it is decided entirely
by the negatives nearest the top.

**Hypothesis:** adding a class-conditional hinge that pushes the `ins_buzz`
logit at least `m` below zero on frames labelled `mech_auto` and not `ins_buzz`
lifts sens@fpr0.005, concentrated at `1_95` and secondarily `mustard` (the other
vehicle-led fold).

**Why not OHEM.** This is deliberately *not* the archived E2 `tail-loss` /
`tail-loss-retest` idea. That term selected its hard negatives by **within-batch
rank**, which (a) made the compiled loss depend on batch composition and so
invalid as the `val_loss` early-stopping signal — the flaw that voided
`tail-loss` — and (b) reproducibly inverted the score ordering at weight 1.0.
The term here is a deterministic per-frame function of `(y_true, y_pred)` with
no rank, no top-k and no batch dependence, so `val_loss` stays a valid stopping
monitor and the archived failure mode does not apply. `mech_auto` is chosen from
this era's own FP census, not from batch statistics.

**Dose, not a point guess.** Run as a two-point dose-response (`lambda` 0.5 and
2.0 at a fixed margin `m=2.0` logits) rather than one blind weight, since a
single-run comparison has an MDE of ~0.027 (`probe-grid`, baseline SD 0.0095
over n=3) and the interesting readout — does `1_95` move at all — is per-fold.

## Changes

## Results

## Conclusion
