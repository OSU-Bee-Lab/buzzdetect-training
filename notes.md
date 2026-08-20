# input-standardization

## Hypothesis

`exp/yamnet-combined` (+0.016) concatenated YAMNet's 521 sigmoid class scores
onto the 1024-d embedding, and the two blocks sit on very different scales
(score block mean 0.004 / max 0.41 vs embedding block mean 0.041 / max 2.6,
roughly 10x). Nothing in `03_train` normalizes its input, and `Dropout(0.2)`
drops units uniformly regardless of scale — the small block may be regularized
far harder than the large one and never get properly used. Standardizing per
dimension before the probe should let the probe use the scores properly, and
decides whether `yamnet-combined`'s gain is a ceiling or an artifact of the
scores being drowned out.

## Changes

`03_train/train.py::_train_one` — one `tf.keras.layers.Normalization(axis=-1)`
layer inserted between `Input` and `Dropout`, `adapt()`-ed on that call's
training folds only (`data.train_tf`, stripped of labels) before `fit()`.

This is not the `BatchNormalization` layer that was a clear negative
pre-CV (-5.8pp, running stats didn't transfer across deployments — see
`IDEAS.md`). `Normalization.adapt()` sets fixed mean/variance once from the
training folds and freezes them as non-trainable weights; nothing updates
during `fit()`. Because it's a layer inside the `Sequential` model, it is
saved into `model.keras` and applied identically at training, validation, and
scoring (including future inference) with no other code touched — unlike
`context-stack`, this needs no inference-wrapper mirroring to be shippable.

No re-extraction: reused the `yamnet_combined` embeddings for `medium`,
already cached from `exp/yamnet-combined`.

    python 03_train/main.py --name yamnet_medium_combined_std --set medium \
        --embedder yamnet_combined --translation general -y

Smoke-tested end to end on the `tiny` set first (11 folds + shipped model,
exit 0) before committing to the medium run.

## Results

Paired against `exp/yamnet-combined` (`models/yamnet_medium_combined`), same
folds, same embedder, only the normalization layer differs:

| fold | yamnet-combined | +standardization | delta | val frames |
|---|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 0.441 | 0.454 | +0.013 | 6908 |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 0.378 | 0.383 | +0.005 | 4710 |
| Lily Adam - One Hive/recorders/willard/2024-08-07/1_11 | 0.204 | 0.217 | +0.013 | 4713 |
| Lily Adam - One Hive/recorders/wooster/2024-07-26/1_143 | 0.296 | 0.307 | +0.011 | 4707 |
| Luke - Diel Drivers/2026-04-08/1_150 | 0.069 | 0.168 | +0.099 | 4891 |
| Luke - Diel Drivers/2026-05-06/1_95 | 0.029 | 0.026 | -0.003 | 6606 |
| Luke - Various Opportunistic Recordings/2025-06-23/1_23 | 0.326 | 0.314 | -0.012 | 314 |
| Luke - Various Opportunistic Recordings/2025-07-03/1_37 | 0.305 | 0.262 | -0.043 | 4712 |
| Luke - Various Opportunistic Recordings/2025-08-05/31 | 0.004 | 0.023 | +0.019 | 942 |
| Luke - Various Opportunistic Recordings/2025-08-12/1_114 | 0.194 | 0.208 | +0.014 | 3768 |
| Luke - Various Opportunistic Recordings/2025-08-27/48 | 0.193 | 0.037 | -0.156 | 1570 |

- sens_persite @ fpr0.005: yamnet-combined 0.222 → this 0.218 (-0.004)
- vs cv-baseline (plain 1024-d, unstandardized): 0.206 → 0.218 (+0.012)
- both runs scored on identical footing: 11/11 folds reached the target FPR,
  same 5641 buzz frames, same median 22 negative frames per threshold
  (`folds_sx.csv` on both).

7 folds up, 4 down. The headline is flat-to-slightly-down, but it is carried
entirely by one fold: `2025-08-27/48` (-0.156, 1570 val frames). That is the
same fold that swung +0.180 in the *other* direction when `yamnet-combined`
was first compared to `cv-baseline` — one of the two `near-chance-deployments`
folds, already flagged there as unreliable per-fold, and below the 3000-frame
line the README uses for "enough buzz to read." Reading it as noise both times
is consistent, not convenient: nothing in this run marks it as a bad draw
otherwise (400/400 epochs, `best_val_loss` in the normal range for this run).

Restricting to the 8 folds with ≥3000 validation frames — the ones worth
reading per fold — gives **6 up, 2 down, mean delta +0.014** (0.240 → 0.253).
The two declines there are small (-0.003, -0.043); five of the six gains are
0.011 or larger, one is +0.099. So on the folds that can be trusted
individually, standardization is a further modest gain on top of
`yamnet-combined`, not a wash — the flat headline is the noisy small fold
pulling the persite mean back down, the same mechanism the README warns about.

**Convergence caveat, worth flagging honestly rather than chasing:** under
standardization, 10 of 11 folds ran the full 400-epoch cap without early
stopping ever firing; the baseline (unstandardized) early-stopped in every
fold, at a median of ~120 epochs. `best_val_loss` is also consistently higher
under standardization (e.g. 0.457 vs 0.417 on `2026-04-08/1_150`) even where
`sens@fpr` improved. This says the fixed Adam LR (0.002) and the
`min_delta=0.002`/`patience=50` stopping rule, both tuned against
unstandardized inputs, are no longer well matched once the input scale
changes — plausible, since a `Normalization` layer changes the effective
gradient scale the optimizer sees. Per LOOP.md, hyperparameters are not this
loop's job absent a strong reason and a structure worth tuning around; this is
recorded as a caveat on the number above; a real read of standardization would
want the LR/patience revisited, not folded into this run's tuning-averse
conclusion.

## Conclusion

Inconclusive-to-mildly-positive, not a wash from the input-scale worry alone:
on the 8 folds worth reading per fold, standardization adds a further +0.014
on top of `yamnet-combined` (6/8 up), so the scale mismatch in the IDEAS
hypothesis was real and the probe does use the scores better once
standardized. But the flat headline (-0.004) and the fact that every fold ran
to the epoch cap under a stopping rule tuned for the old input scale mean this
isn't a demonstrated win either — the training dynamics changed in a way this
run didn't control for, and the single volatile small fold is doing as much
work as the standardization itself.

Net read: `yamnet-combined`'s +0.016 was not obviously an artifact of the
scores being drowned out — standardizing the input didn't unlock a bigger gain
on the metric that counts every deployment once, and the honest per-fold read
is a small additional gain from a change that also broke the epoch-budget
assumptions the pipeline's defaults were tuned under. Not worth adopting on
this evidence. If this is revisited, fix the convergence question first
(retune LR/patience for standardized inputs in the one deliberate sweep
LOOP.md allows) before trying to read the sens number again.
