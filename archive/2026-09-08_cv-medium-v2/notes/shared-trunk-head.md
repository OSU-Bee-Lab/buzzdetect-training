# shared-trunk-head

## Hypothesis

`train.py` builds `Input -> Dropout(0.2) -> Dense(n_classes)`: one layer, no
hidden stage, no activation, into `weighted_cross_entropy_with_logits`. So
`ins_buzz`'s logit is a function of `W[:, buzz]` and `b[buzz]` alone and the
15 classes are **15 decoupled logistic regressions** sharing only the input
dropout mask ([[decoupled-probe-head]]). No gradient path runs from a
`mech_auto` error to `W[:, buzz]`.

Insert a shared ReLU hidden layer — `Dropout -> Dense(h, relu) -> Dropout ->
Dense(n_classes)` — and all 15 heads read one learned representation, so
auxiliary-class supervision shapes features the buzz neuron also uses. This is
the **multi-task-transfer** claim, which is a different question from the
capacity claim MLP heads were closed on in E1/E2 ("does the probe need more
capacity?" — no). Nobody has asked whether the auxiliary classes should inform
buzz at all.

This is step (1) of IDEAS.md's two-step sequencing: **hidden layer alone,
uniform loss**, against `cv_baseline`. Per-neuron buzz weighting on top is step
(2) and only runs if (1) is not a clear negative — running them together
confounds an architecture change with a loss change.

Priors are genuinely mixed and this is not a favourite:

- Against: YAMNet's 1024-d is already linearly separable for this concept by
  construction (it is the penultimate layer of a supervised classifier whose
  AudioSet vocabulary contains `Buzz` and `Bee, wasp, etc.`) — exactly the case
  where a hidden layer buys least. The pool is ~72k frames with ~7.7k buzz, so
  a wide hidden layer can overfit the training sites.
- For: `mech-margin` (2026-09-09) attacked the confuser problem directly with a
  class-conditional margin and failed monotonically across a 64x dose ladder,
  and its mechanism was that a **linear** readout of frozen YAMNet cannot push
  `mech_auto` frames down without taking buzz with them — despite
  `cosine(W_ins_buzz, W_mech_auto) = -0.016`, i.e. near-orthogonal readouts
  overlapping in the *frame* population. That is precisely the situation where
  a non-linear stage has something to add.

**Run as a width ladder, not a single point.** `probe-grid` put the minimum
detectable effect of a single-run comparison at ~0.027 (baseline SD 0.0095 over
n=3), so one run at one width cannot distinguish a small real effect from a
draw. Three widths — h = 64 / 256 / 1024 — give a dose-response reading the way
`mech-margin`'s ladder did: a real multi-task effect should be non-flat and
ordered, and overfitting (if that is what happens) should worsen with width.

## Changes

`--hidden N` on `03_train/main.py`, threaded to `_train_one`. Default `0` =
the shipped decoupled head, byte-identical code path. Nothing else changes:
same loss (`weighted_bce_loss`, label smoothing 0.2), same Adam 0.002, same
`RestoreTrueBest` on `val_loss`, same dropout rate on both stages, `medium` /
`yamnet` / `general`. Branched from main, so **no `--monitor val_sens`** —
`probe-grid`'s flag is unmerged and adopting it is Luke's call, so the
comparator is `cv_baseline` as logged.

## Results

Baseline `cv_baseline` = 0.218. Ladder: **h=64 → 0.219, h=256 → 0.221,
h=1024 → 0.226** (+0.001 / +0.003 / +0.008).

| fold | buzz_frames | base | h=64 | h=256 | h=1024 |
|---|---|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 2144 | 0.426 | **+0.033** | **+0.025** | **+0.034** |
| Lily - Fit+Fast/.../53 | 1031 | 0.425 | **-0.043** | **-0.023** | **-0.007** |
| Lily Adam - One Hive/willard/1_11 | 305 | 0.180 | +0.017 | 0.000 | +0.013 |
| Luke - Diel Drivers/2026-04-08/1_150 | 146 | 0.021 | +0.006 | +0.013 | +0.006 |
| Luke - Diel Drivers/2026-05-06/1_95 | 433 | 0.037 | -0.005 | +0.002 | -0.007 |
| **mean** | | **0.218** | **0.219** | **0.221** | **0.226** |

Up/down: 3/2, 3/1 (1 flat), 3/2. Every delta is inside `probe-grid`'s ~0.027
single-run MDE, and the largest headline (0.226) sits inside the *baseline's
own* measured range (0.208 / 0.218 / 0.227 over n=3, SD 0.0095). Nothing here
is resolvable as an effect.

**The one thing that is not noise-shaped**: the two rich folds move
consistently and in *opposite* directions at all three widths — 1_29 gains
~+0.03 every time, Fit+Fast loses every time. Both are well above the ~0.019
repeat-movement `harmonic-comb` measured for >1000-buzz folds, and neither
flips sign across three independent runs, so this reads as a real trade rather
than scatter. The mean is null because the trade nets out. Widening shrinks the
Fit+Fast loss (-0.043 → -0.007) while 1_29 holds, which is why the headline
drifts up monotonically without any fold actually improving.

**1_95 and 1_150 — the folds this was aimed at — did not move.** 1_95 is
-0.005 / +0.002 / -0.007. That is the direct test of the mechanism: `mech-margin`
failed because a *linear* readout cannot push `mech_auto` frames down without
taking buzz with them, and the hypothesis was that a non-linear stage could.
Given one, it does not.

**Convergence diagnostic** (mean over folds):

| model | best_epoch | best_val_loss |
|---|---|---|
| cv_baseline | 114.8 | 0.9101 |
| hidden_64 | 109.8 | 0.9064 |
| hidden_256 | 60.8 | 0.9074 |
| hidden_1024 | 46.6 | 0.9075 |

The trunk reaches the *same* val_loss (4th decimal) in progressively fewer
epochs — 2.5x fewer at h=1024. Same destination by a shorter path, the same
signature `combined-revalidate` found for the AudioSet sigmoid block. The
hidden layer is fitting the objective faster, not fitting it better, which is
what a null on a representation that is already linearly separable should look
like.

## Conclusion

**NULL. Do not build on it.** +0.001 / +0.003 / +0.008 across a 16x width
ladder, folds split 3/2 at every width, everything inside the MDE and inside
the baseline's own draw-to-draw range. Not a clear negative either — no width
hurt — but there is no gain here to bank.

The hypothesis is answered on its own terms: connecting the 15 classes through
a shared representation is *available* to help and does not. `mech-margin`'s
mechanism suggested a non-linear stage was the missing ingredient for the
confuser problem; given one, 1_95 stays put at every width. Combined with the
convergence diagnostic (same val_loss, fewer epochs), the reading is that
YAMNet's 1024-d code already carries what a shared trunk could learn — the
`Dropout -> Dense` head was not the constraint.

**Step (2) of the IDEAS sequencing (per-neuron buzz weighting on top of the
trunk) is formally unblocked but no longer motivated.** Its whole premise was
that weighting needs something to be weighted *against*; the trunk provides
that and buys nothing, so the follow-up would be tuning a loss on a structure
with no demonstrated headroom. LOOP.md's "err against hyperparameter tuning"
applies. Left in IDEAS as explicitly deprioritised rather than closed.

The genuinely interesting residue is the 1_29-up / Fit+Fast-down trade, stable
across three widths. Both are rich folds, so it is measurable; nobody has asked
what distinguishes them. That is a diagnostic, not an architecture run, and
diagnostics are the category LOOP.md says survives a data change.

`--hidden` is kept in the code, defaulting to 0 (the shipped head), the same
way `--standardize` was kept after `standardize-blocks`.
