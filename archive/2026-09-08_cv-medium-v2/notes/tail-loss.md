# tail-loss

## Hypothesis

Every logged CV-era model trains with plain
`BinaryCrossentropy(label_smoothing=0.2)`, which spends its gradient across
the whole score distribution. But the endpoint (`sensitivity_mean @
fpr0.005`) reads a narrow slice: `models/yamnet_medium_general/folds_sx.csv`
shows `neg_frames_fold_median = 22` — a typical fold's threshold rests on
roughly twenty top-scoring negative frames out of several thousand. BCE has
no mechanism that knows those twenty frames are the ones that decide the
metric; it treats a trivially-easy negative the same as a near-threshold one.

Adding a hard-negative-mining (OHEM-style) term that specifically upweights
the highest-scoring negative frames on the `ins_buzz` neuron, within each
training batch, should push those borderline negatives down and lift
sensitivity at low FPR, without needing to touch the bulk of the easy-negative
mass BCE already handles fine.

This is not a rerun of `focal-loss` (pre-CV dead end, "shifts the operating
point, doesn't lift the curve"): focal reweights every example by the model's
own confidence, uniformly across the whole score range. OHEM reweights by
*rank position among negatives in the batch*, which is a different
mechanism — it's blind to how confident the model is on a given hard negative,
only to whether that negative is currently in the top slice. Will call out in
Results if the effect looks focal-shaped anyway (a pure operating-point shift
rather than a curve lift).

## Changes

Custom Keras loss `TailBCELoss` added in `03_train/train.py`, replacing the
`model.compile(loss=...)` call in `_train_one`. Formulation:

- Base term: elementwise sigmoid cross-entropy with label smoothing 0.2,
  applied to all classes, averaged over classes per sample — numerically the
  same as `BinaryCrossentropy(from_logits=True, label_smoothing=0.2)`'s
  per-sample output.
- OHEM term: within each training batch, take the `ins_buzz` neuron's
  logits for frames labelled negative on `ins_buzz`. Select the top
  `ohem_frac` fraction of those by logit (highest-scoring negatives — the
  ones nearest or above the eventual operating threshold). Compute their
  smoothed BCE against the negative target and scatter-add
  `ohem_weight * that loss` back onto each selected sample's per-sample loss
  vector (all other samples get +0).
- Returned per-sample loss vector has the same shape/semantics as
  `BinaryCrossentropy`'s `call()` output, so Keras's `sample_weight`
  machinery (i.e. `class_weight=` as passed to `fit()`) multiplies and
  reduces it exactly the same way as the baseline loss — including the
  known `argmax`-collapse bug from `class-weight-fix`. Not fixing that bug
  here; keeping it live is what makes this comparable to `cv-baseline`.
- `ohem_weight` fixed at 1.0 (equal weight to the base BCE term).
  `ohem_frac` is the one hyperparameter varied: fraction of a batch's
  negatives (by `ins_buzz` score) that count as "hard" each step.

Training batches are close to one shot per epoch (`size_batch=65568` vs.
~70k frames per fold's training pool), so "top fraction of negatives in the
batch" is close to "top fraction of negatives in the whole training pool"
each epoch — the same scale the validation threshold is drawn from, just ~10x
larger since the training pool spans ~10 folds' negatives.

## Results

`smoke_test_loss` passed (`TailBCELoss` compiles, fits one step, round-trips
`save(include_optimizer=True)` -> `keras.saving.load_model(compile=False)`,
predicts finite output) — see `tools/diag_tailloss.py` and the smoke run in
the session log. That only exercises small fixed-shape dummy data, though;
it did not catch what the real run below hit.

Per LOOP.md, diagnosed on 2 folds before spending a full CV, using
`tools/diag_tailloss.py` (not part of the pipeline, not for the CV — trains
one fold at a time via the same `_load_data`/`_train_one` used by `train_set`,
scores it, and stops). `ohem_weight` fixed at 1.0; `ohem_frac=0.01` (top 1%
of a batch's `ins_buzz` negatives, chosen to match the training pool's scale
to the ~0.5%-FPR region the metric reads — the training pool spans ~10 folds
vs. one at validation, so the equivalent top-count is roughly 10x the
validation-side ~22-frame median).

| fold | baseline sens@fpr0.005 | baseline best_epoch | this exp sens@fpr0.005 | this exp best_epoch | val frames |
|---|---|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 0.448 | 63 | 0.011 | 17 | 6908 |
| Luke - Various Opportunistic Recordings/2025-08-12/1_114 | 0.163 | 129 | 0.003 | 6 | 3768 |

2-fold pooled sensitivity_mean (same code path as `folds_sx.csv`, run over
just these 2 folds): baseline 0.305 -> this exp 0.007. Both folds down,
by an order of magnitude, both on val-frame counts large enough to trust
individually (>3700).

**Diagnosis, not just a number.** Both folds' loss curves show the same
shape: `val_loss` drops to a minimum in the first handful of epochs, then
climbs steadily for the rest of training even as train accuracy keeps
improving. That's because `TailBCELoss` is the *compiled* loss, so Keras
computes it — OHEM term included — on the validation batch too, and that
term is `EarlyStopping`'s `val_loss` monitor. The OHEM term picks the current
top-1%-by-score negatives *within whatever batch it's given*; on the single
held-out fold that's a much smaller, noisier population than the pooled
training batch, so which frames count as "hard" swings from epoch to epoch
early in training when the model's ranking is still close to random. That
produces a spurious early minimum, and once training moves past it,
`val_loss` trends only upward — patience=50 lets it run a while longer, but
`restore_best_weights` locks in the near-random early snapshot (best_epoch 6
and 17, vs. baseline's 129 and 63 on the identical folds).

A second, independent problem surfaced at `ohem_frac=0.01, ohem_weight=0.1`
(lower magnitude, same folds): the same rising-val_loss shape appeared, and
training then crashed outright with `InvalidArgumentError: indices[...] is
not in [0, N)` inside `tail_bce_loss/cond/GatherV2`, during evaluation.
`top_k`'s `k` is data-dependent (a fraction of however many negatives are in
this batch), and that dynamic shape does not survive Keras's fused
multi-step training/eval execution cleanly on this backend — a real
implementation fragility on top of the monitor problem, not something a
different hyperparameter value fixes.

Given two large, reliable folds both collapsed by an order of magnitude for
an identifiable structural reason (not small-fold noise — the
overfitting-the-tail check the assignment asked for doesn't even apply here,
since both diagnostic folds are large and the failure isn't fold-size
correlated, it's monitor-composition correlated), a full 11-fold CV was not
run. Spending it would not plausibly reverse this.

## Conclusion

Null/negative result, but not a clean read on the hypothesis itself — the
diagnostic exposed an implementation defect (the OHEM term contaminating the
`val_loss` early-stopping monitor, plus a graph-execution bug from the
data-dependent top-k) rather than a clean test of "does tail-targeting lift
sensitivity." The failure mode does not look like `focal-loss`'s
("shifts the operating point, doesn't lift the curve") — this isn't a shifted
operating point, it's undertrained models produced by early stopping firing
on a corrupted signal. So this doesn't reproduce the focal-loss verdict; it's
a different, structural failure specific to folding a rank-dependent term
into the same loss object used for both training and the stopping monitor.

The idea itself is not ruled out by this: a version that computes the OHEM
term only during training (e.g. monitor plain BCE for early stopping, or
compute the tail term over a fixed reference pool rather than whatever's in
the current eval batch) might behave completely differently. But that's a
different, more careful implementation and a separate experiment — not a
hyperparameter retry of this one. Given LOOP.md's guidance to err against
tuning and this loop's compute budget, stopping here rather than iterating
on the monitor design within this experiment.
