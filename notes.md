# aves-readout

## Hypothesis

`aves-probe` (2026-09-09) came out 0.218 -> 0.074, 5 folds down. But the
geometry diagnostic run alongside it says the buzz information **is in the
embeddings**: AVES's mean per-dimension separation is the *better* of the two
(mean |Cohen d| 0.242 vs YAMNet's 0.155; mean |AUC-0.5| 0.074 vs 0.022). Only
where signal concentrates does YAMNet win (top-10 dims 0.260 vs 0.226).

So this is a **readout failure, not an ignorance failure**, and the failure mode
is named: 768 dense, weakly-informative, zero-centred dimensions, read by a
linear probe that must generalise to a held-out *site*. The probe can fit the
training folds from many weak site-correlated directions that don't transfer.
The training curve is the signature — `1_29`'s val_sens peaks at 0.0020 in
epoch 3 and decays to 0 while train loss keeps falling.

**Question: can the information be recovered without re-extracting anything?**

Three input-side transforms, all fit on training folds only, all pure functions
of embeddings already on disk:

1. **Standardize** per dimension. The probe's first layer is `Dropout(0.2)`
   applied *directly to the input*. On YAMNet's 89.6%-zero non-negative code
   that is mild. On a dense signed code of heterogeneous per-dimension scale it
   is heavy multiplicative noise, and scale alone decides each dimension's
   gradient share. Implementation already exists, fold-safe and
   save/load-clean, on `exp/standardize-blocks`.
2. **PCA to k dims**, whitened. Attacks the diagnosis directly: concentrate a
   distributed code into few directions and cut the probe from 768xC to kxC
   free parameters, shrinking the cross-site overfitting surface.
3. **Regularisation strength** appropriate to a dense code (L2 / dropout rate).

### Why the archived dead ends don't already answer this

`LOOP.md` lists MLP heads and L2 among the conclusions likeliest to still hold.
Both were measured **on YAMNet**, and both notes say so in their own words —
`mlp-head-repro`: "buzz is already linearly accessible in *this space*, and
extra capacity doesn't add transferable signal"; `l2-regularize`: "Dropout(0.2)
+ label_smoothing=0.2 already provide sufficient regularization" for *this
architecture*. Those are statements about a sparse, concentrated,
already-linearly-separable representation. The whole point of the geometry
table is that AVES's is the opposite kind. A regulariser that over-regularises
a code where the answer is already written down linearly is not evidence about
a code where it is smeared across 768 weak directions.

That is a reason to re-ask the question in the new space, not a reason to expect
a different answer. If these come back negative too, the honest reading is that
the readout is not the problem and AVES is simply worse here.

### Method: diagnose offline before spending a CV

Every transform above is a numpy operation on cached embeddings, so the sweep
runs **offline against `03_train/metrics.py:sens_at_fpr` directly** — same
metric, same leave-one-fold-out rotation, no Keras, seconds per configuration
instead of ~9 min. Only a configuration that clears `cv_baseline` offline earns
a real CV run for a loggable number.

YAMNet is swept identically as a control. If a transform lifts AVES and leaves
YAMNet flat, that is the geometry-specific effect the hypothesis predicts; if it
lifts both, it is a probe-config finding that was never about AVES.

**Expectation:** partial recovery. Enough to say whether the representation is
usable, not enough to beat 0.218 — that would take the non-linear head this
run deliberately does not touch (standing 1-layer injunction).

## Changes

None to `03_train`. The whole run is offline analysis over the embeddings
`aves-probe` already wrote to the shared cache:

- `sweep_cache.py` — dumps per-fold `(X, y)` matrices for all 44 folds
  (39 `train-only` + 5 `rotate`), both embedders, via `build_fold_dataset` and
  the `general` translation, so a configuration doesn't re-read 33k pickles.
  Buzz counts reproduce `folds_sx.csv` exactly.
- `sweep.py` — leave-one-fold-out over the 5 rotating folds, scored through
  `03_train/metrics.py:sens_at_fpr` at FPR 0.005, head =
  `sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=2000,
  class_weight='balanced')`. 35 configs per embedder: `{raw, std, pca-k for k in
  16..256} x {C in 1e-3..10}`. YAMNet swept identically as control.

## Results

### Headline

| | shipped Keras probe | best offline linear | best config |
|---|---|---|---|
| AVES 768-d | 0.074 | **0.194** | `std`, C=1e-2 |
| YAMNet 1024-d | 0.218 | **0.261** | `raw`, C=1e-1 |

Same embeddings, same folds, same metric, same rotation. Only the head differs.

### The hypothesis in the Hypothesis section is wrong on mechanism

I wrote "the probe overfits a distributed representation across sites." Three
things falsify that, and they were all checkable before the run:

1. **The training pool is 43 folds, not 5.** 39 `train-only` idents plus the 4
   non-held-out rotations: ~72k frames, ~7.7k buzz frames, 37 of 39 train-only
   folds carrying buzz. Site scarcity is not the problem; the probe sees more
   site diversity than the 5-fold rotation suggests.
2. **`val_sens` never decayed.** Recovered from `exp/aves-probe`'s training log,
   per fold: best `val_sens_fpr0.005` occurs at or within two epochs of the
   **final** epoch in **all five folds** — 0.0166@153/153, 0.2159@76/77,
   0.1343@78/78, 0.0347@50/67, 0.0196@131/133. Monotone improvement to the end.
   My earlier "peaks at epoch 3 and decays to 0" was the first ~17 epochs of a
   153-epoch run read as a trajectory. It is start-up transient.
3. **The probe is trained essentially full-batch.** `train.py:132` sets
   `size_batch = 65568` against ~72k training frames — **2 gradient steps per
   epoch**. `1_29` stopped at epoch 153, so the whole fit was ~306 Adam steps.

**The AVES probe was undertrained, not overfit.** `EarlyStopping` monitors
`val_loss` (`patience=50, min_delta=0.002, restore_best_weights=True`), and
training halts on a `val_loss` plateau while `val_sens` is still climbing —
`1_29`'s `val_loss` bottoms at epoch 131 and the run dies at 153 with `val_sens`
still rising. `lbfgs` at `max_iter=2000` simply converges, which is most of the
0.074 -> 0.194.

This also explains the otherwise odd shape of the C sweep: **regularisation
strength barely matters.** AVES spans 0.143-0.193 across four orders of
magnitude of C and YAMNet spans 0.253-0.261. AVES sits at 0.143 at the *most*
constrained setting, already +0.069 over the Keras probe before any tuning.
Nothing about the penalty explains the gap; convergence does.

### `restore-on-sens` is the archived lever this needs, and its caveat doesn't apply

`archive/2026-08_cv-medium-v1/notes/restore-on-sens.md` is caveated-positive
(+0.014, 9 up / 0 down) and was shelved because "the frozen probe's `val_loss`
barely diverges from sens — the divergence is a backbone-fine-tuning
phenomenon." **That is a YAMNet statement.** On AVES the frozen probe diverges
hard: `val_loss` flat from epoch 131, `val_sens` still climbing at 153. Same
pattern as the `mlp-head`/`L2` dead ends — a verdict measured in YAMNet's
geometry, silently inherited.

### The geometry result, which does survive

PCA-whitening to k dimensions, fit on training folds only. This was meant to
*help* AVES by shrinking the parameter count; it did the opposite, and in doing
so measured the sparse-vs-distributed claim directly:

| dims kept | YAMNet | (% of its best) | AVES | (% of its best) |
|---|---|---|---|---|
| 16 | 0.235 | 90% | 0.055 | 28% |
| 32 | 0.246 | 94% | 0.085 | 44% |
| 64 | 0.248 | 95% | 0.114 | 59% |
| 128 | 0.250 | 96% | 0.132 | 68% |
| 256 | 0.242 | 93% | 0.182 | 94% |
| all | 0.261 | 100% | 0.194 | 100% |

YAMNet keeps 90% of its sensitivity in **16** principal directions. AVES needs
~256 to reach 94% and collapses to 28% at 16. This is independent of the
training-procedure confound above — every row is a converged `lbfgs` fit — so it
stands on its own as the quantitative form of "YAMNet's buzz evidence is
concentrated, AVES's is distributed."

Standardisation matters only where the fit is constrained (AVES `raw` C=1e-3
0.143 vs `std` C=1e-3 0.187); at each embedder's best C it is worth ~0.001 and
YAMNet mildly prefers `raw`.

### AVES is still worse under a matched readout

0.194 vs 0.261 overall, and 0.229 vs 0.468 on `1_29`, the one buzz-rich fold.
The direction in `aves-probe` was right. The **magnitude was inflated ~2.5x** by
a head that fails to converge on this input.

## Conclusion

**Most of `aves-probe`'s -0.144 is the training procedure, not the embedder.**
Under converged linear readouts AVES is 0.194 against YAMNet's 0.261 — worse,
clearly and on the fold that can resolve it, but not the 0.074 collapse that was
logged. `log.jsonl`'s `aves-probe` entry is amended to `caveated` accordingly.

**The transferable finding is about the pipeline, not about AVES.** The shipped
probe leaves ~0.043 on the table for YAMNet too (0.218 -> 0.261) — that is
`cv_baseline`, the comparator every number in this era is measured against. The
mechanism is identified but not yet attributed to a single lever: full-batch
Adam at ~2 steps/epoch, `EarlyStopping` on `val_loss` rather than the shipped
metric, `min_delta=0.002`, `Dropout(0.2)` on the input, `label_smoothing=0.2`,
15-class multi-label vs binary. The offline sweep cannot separate these because
it changes all of them at once. Each is now a separate line in `IDEAS.md`.

**Do not read the offline numbers as pipeline numbers.** `lbfgs` +
`class_weight='balanced'` + binary is not `weighted_bce_loss` + label smoothing
+ Adam. 0.261 is not a claim that `cv_baseline` is really 0.261; it is a claim
that a converged linear readout of the same embeddings reaches it, which is
evidence that the gap is reachable, not that any particular flag will reach it.

**Cost note for whoever grids this.** These are frozen-probe CVs (~9 min each)
and the offline sweep is ~40 min for 70 configurations, so the whole lever grid
is affordable in a way a fine-tune is not.

**Method note.** The offline-sweep-before-verdict habit is now written into
`LOOP.md` under Constraints; this run is what it was written from. It cost 40
minutes and moved a logged verdict by 2.5x.
