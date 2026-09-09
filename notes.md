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

## Conclusion
