# standardize-blocks

## Hypothesis

`combined-revalidate` measured the thing this experiment is about: on
`yamnet_combined`'s 1545-d input, the 1024-d YAMNet block runs **~8x** the
sigmoid block in mean per-dim sd (0.089 vs 0.011) and **~13x** in magnitude,
with one shared Adam learning rate serving both. A linear probe under a single
LR cannot give a 10x-smaller block a proportionate share of the gradient, so the
521 AudioSet scores are present but underweighted — the archived note's
"nothing normalizing the two", now measured on live data.

Per-dim standardization of the input removes the scale difference. On the
pre-revision data it was worth **+0.014 on top of combined (6/8 folds)** and was
not adopted only because 10/11 folds then ran the full 400-epoch cap against a
median ~120 — the LR and patience were tuned for the old input scale. That is a
convergence-speed problem, not a verdict, and it is the thing to handle head-on
here rather than to be surprised by.

Note what is and is not being claimed. The 521 scores are a supervised readout
of the 1024-d block beside them, so they carry no new audio information
(`combined-revalidate`: same `best_val_loss` in fewer epochs). The claim is only
that a badly-scaled block cannot contribute even the redundant-but-conveniently-
shaped part of what it has.

## Changes

`03_train`, behind a new `--standardize` flag (off by default; the control is
the same code path with the flag off):

- `_input_stats(samples)` — per-dim mean/variance of the training pool,
  streamed in float64. **Fold-safe by construction:** it is handed the training
  samples only, never the held-out fold, so the transform applied at validation
  and scoring time is fit strictly inside the training pool. Same rule CLAUDE.md
  states for augmentation, same reason.
- A `Normalization` layer as the **first layer of the model**, with mean and
  variance fixed at construction. `adapt()` is never called, so the layer cannot
  pick anything up from validation data; and because it lives in the model,
  `_score_fold` and the shipped inference path apply the identical transform
  without knowing it exists.
- **`VAR_FLOOR = 1e-6`, the known bug this idea carries.** A dim whose training
  variance is below it is passed through unchanged (mean 0, variance 1) instead
  of being scaled. Keras divides by `sqrt(var + 1e-7)`, so a dim that never
  varies gets amplified ~1e4x into pure noise — the NaN blowup the archived
  attempt hit. **554 of 1545 dims** hit the floor on a real fold: dead AudioSet
  classes that never fire on this corpus, plus dead YAMNet dims.
- `n_dims_passthrough` and `standardize` recorded in each fold's `summary.json`
  and `config_model.json`.

Verified before running: `Normalization` survives the
`save(include_optimizer=True)` -> `load(compile=False)` round trip the inference
path uses (max output diff 0.0, weights preserved), and a floored dim is exactly
the identity.

### The epoch cap, handled up front

A one-fold probe (1_150, the thinnest fold) confirmed the archived symptom
immediately: standardized, it ran to the **400-epoch cap** with its best epoch
at **393**, where the unstandardized control stops at 66. Standardization
changes the effective learning rate, so the default cap is binding for it and
not for the control — comparing there would compare a converged model against a
truncated one.

So the main run raises `--epochs` to 3000 with the **stopping rule untouched**
(`patience` 50, `min_delta` 0.002 on `val_loss`). The cap is a mechanical guard,
not a tuned hyperparameter, and it is non-binding for the control at ~130-170
epochs; making it non-binding for both sides is what makes this one comparison
of one change, rather than a comparison of a model that finished against one
that ran out of budget. If the standardized run still hits 3000, that is
reported as a failure to converge, not as a number.

Control: `combined_control`, same embedder, same data, flag off, retrained here
because `combined-revalidate`'s model directory went with its pruned worktree.

## Results

## Conclusion
