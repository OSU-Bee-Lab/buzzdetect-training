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

<!-- filled in as the run goes -->

## Results

<!-- filled in when the run lands -->

## Conclusion

<!-- filled in when the run lands -->
