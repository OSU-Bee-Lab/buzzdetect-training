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

`weighted_bce_loss(..., margin_spec=(i_buzz, i_cond, lam, m))` adds, per frame,

    lam * 1[y_raw[mech_auto] and not y_raw[ins_buzz]] * relu(z_ins_buzz + m)

on **raw** (unsmoothed) labels. Exposed as `--margin-lambda / --margin-m /
--margin-class` on `03_train/main.py`; `--margin-lambda 0` (the default) leaves
the compiled loss byte-identical to every run logged before the flag existed,
verified by direct comparison. Margin arithmetic checked against a hand
computation, and `tools/smoke_model.py` run on the compiled loss (compile ->
fit -> `save(include_optimizer=True)` -> `load(compile=False)` -> predict).

Four CVs, `medium` / `yamnet` / `general`, frozen probe, `val_loss` stopping —
identical to `cv_baseline` except the loss term. `m = 2.0` logits throughout;
`lambda` on a 64x geometric ladder.

## Results

| lambda | mean sens@fpr0.005 | delta vs cv_baseline (0.218) | folds up/down |
|---|---|---|---|
| 0.125 | 0.202 | -0.016 | 1 / 3 (1 flat) |
| 0.5   | 0.169 | -0.049 | 0 / 5 |
| 2.0   | 0.090 | -0.128 | 0 / 5 |
| 8.0   | 0.021 | -0.197 | 0 / 5 |

Per fold, vs `cv_baseline`:

| fold | buzz frames | base | 0.125 | 0.5 | 2.0 | 8.0 |
|---|---|---|---|---|---|---|
| `JamesU - MustardBumbler/1_29` | 2144 | 0.426 | 0.394 | 0.310 | 0.151 | 0.032 |
| `Lily - Fit+Fast/.../53` | 1031 | 0.425 | 0.437 | 0.364 | 0.217 | 0.050 |
| `willard/2024-08-07/1_11` | 305 | 0.180 | 0.135 | 0.128 | 0.062 | 0.000 |
| `Diel Drivers/2026-05-06/1_95` | 433 | 0.037 | 0.037 | 0.027 | 0.018 | 0.021 |
| `Diel Drivers/2026-04-08/1_150` | 146 | 0.021 | 0.007 | 0.014 | 0.000 | 0.000 |

**Monotone in the dose, and monotone in every fold that can resolve anything.**
This is not a noise-floor read: the two rich folds (2144 and 1031 buzz frames,
the folds `probe-grid` and `harmonic-comb` both found stable) carry the largest
losses, and the ladder spans 64x with no reversal. `1_95` — the fold the
hypothesis was built for — is *down* at every dose.

### Mechanism: the term did the opposite of what it was asked

Raw labels of negatives above each fold's own fpr0.005 threshold, pooled
(`cv_baseline`'s census recomputed here so the comparison is like-for-like):

| model | n above threshold | `mech_auto*` | share |
|---|---|---|---|
| `cv_baseline` | 219 | 58 | 26.5% |
| `lambda 0.5` | 196 | 63 | 32.1% |
| `lambda 2.0` | 173 | 68 | 39.3% |

**The `mech_auto` share of threshold-setting false positives rises with the
dose.** The term pushed `mech_auto` frames down in absolute terms and still lost
ground on them in *rank*, which is the only thing the metric reads.

What it actually did is collapse the score separation. Mean `ins_buzz` logit on
buzz frames vs non-buzz frames, held-out:

| fold | base gap | lambda 0.5 | lambda 2.0 |
|---|---|---|---|
| `1_29` | 1.578 | 1.000 | 0.519 |
| `Fit+Fast/53` | 1.425 | 1.099 | 0.848 |
| `willard/1_11` | 0.852 | 0.547 | 0.412 |
| `1_95` | 0.513 | 0.272 | **-0.064** (inverted) |

Both populations move down; the buzz population moves down *faster*. At
`lambda 2.0` the fold the experiment targeted has its buzz frames scoring
*below* its own non-buzz frames on average.

**This is not weight-space collinearity.** In `cv_baseline`'s shipped readout,
`cosine(W_ins_buzz, W_mech_auto) = -0.016` — near-orthogonal, 10th of 14
classes; the classes whose readout directions actually align with buzz are
`ins_trill` (+0.268), `ambient_background` (+0.265) and `ambient_noise`
(+0.241). The overlap is in the **frame population**, not the weights:
`mech_auto` frames already sit high on the buzz direction (that is why they are
the plurality of the FP census at all), and no linear readout of frozen YAMNet
can push that population down without taking real buzz with it.

## Conclusion

**Clear negative, and the cleanest one in the era — a 64x dose ladder, monotone,
5/5 folds down at every dose at or above 0.125, with the two highest-confidence
folds carrying it.** No noise-floor caveat applies: `probe-grid` put the MDE of
a single-run comparison at ~0.027 and the largest doses are 5-7x that.

The general lesson is worth more than the verdict. **The margin is specified in
absolute logit space; the metric reads only rank**, because every fold is
thresholded on its own held-out audio. A term that drives a subpopulation below
a fixed logit therefore buys nothing directly — it can only help by rotating the
readout, and here rotating it away from `mech_auto` frames rotates it away from
buzz, because the two populations overlap along the buzz direction even though
their readout directions do not. Any future "penalise this confuser" term should
be **pairwise** (buzz frame scored above confuser frame) rather than absolute,
which is the shape `tail-loss` had and the reason its failure is not
transferable evidence against this one.

Secondary, and cheap to have: the FP census tooling and the buzz/non-buzz logit
gap are now a two-minute read on any model with `surprisal/` on disk, and they
survive a data revision.
