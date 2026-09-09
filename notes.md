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

All 5 folds converged well inside the raised cap — 502-1000 epochs against the
3000 guard, best epoch always within ~1% of the last. The cap is not binding,
so this is a comparison of two converged models. (It also confirms the probe's
warning was real: the control stops at 19-160, the standardized run needs
500-1000. Standardization costs ~6x the epochs.)

Paired against `combined_control`, sens@FPR 0.005:

| fold | buzz frames | control | standardized | delta |
|---|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 2144 | 0.436 | 0.436 | 0.000 |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 1031 | 0.424 | 0.421 | -0.003 |
| Lily Adam - One Hive/.../willard/2024-08-07/1_11 | 305 | 0.223 | 0.236 | +0.013 |
| Luke - Diel Drivers/2026-05-06/1_95 | 433 | 0.032 | 0.027 | -0.005 |
| Luke - Diel Drivers/2026-04-08/1_150 | 146 | 0.034 | 0.185 | +0.151 |

Headline **0.230 -> 0.261 (+0.031)**, 2 up / 2 down / 1 flat.

## Conclusion

**Negative, despite a positive headline.** The entire +0.031 is one fold:
1_150, +0.151. Drop it and the mean delta is +0.001 across the other four.

1_150 is the fold the handoff flagged as untrustworthy in advance — 146 buzz
frames, and it moved 0.007 -> 0.062 between two *identical* runs earlier in
this era. A +0.151 swing on it is inside the behaviour that fold has already
demonstrated without any code change, and the pre-registered reading was that a
real effect has to show on the two rich folds. It does not: 1_29 is 0.000 and
Fit+Fast is -0.003, both far inside the 0.014-0.016 noise floor, and both are
the folds where the probe has enough buzz frames to resolve a 3% change.

So the hypothesis is not supported at the scale it predicted. The scale
mismatch between the two blocks is real and measured (~8x sd), and
standardization does remove it, but on the rich folds the probe's headline is
unchanged by removing it — consistent with the hypothesis's own caveat that the
521 sigmoid scores carry no new audio information, only a differently-shaped
readout of the block beside them. Fixing their gradient share buys nothing
because there was nothing extra there to gain.

Not adopting `--standardize`. The flag stays in, off by default: it is
fold-safe, correct, round-trips through save/load, and costs ~6x epochs for no
measured gain, so it is available if a future embedder genuinely mixes
heterogeneous blocks rather than a block and its own readout.

The one thing worth carrying forward is not about standardization: **1_150 and
1_95 are not measuring anything.** Both sit at 0.03 in the control and both
have swung by more than any effect this era has produced. Two of five folds
contributing pure noise to a 5-fold mean is why the headline moved +0.031 on a
null result. That belongs in IDEAS as a fold-weighting or thin-fold-exclusion
question, not as another embedder experiment.
