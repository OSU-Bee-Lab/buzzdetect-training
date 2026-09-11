# combined-revalidate

## Hypothesis

`yamnet-combined` is the second of the two archived clean embedder wins
(`context-embedder` being the first, revalidated 2026-09-08 at +0.040). It
concatenates YAMNet's 1024-d penultimate embedding with the 521 AudioSet
sigmoid class scores from the head that sits on top of it — 1545-d in all. The
521 scores are a nonlinear, supervised readout of the same 1024-d
representation, so they carry no new audio information, but they carry
*AudioSet's* decision boundaries (`Insect`, `Bee, wasp, etc.`, `Buzz`,
`Engine`, `Vehicle`) as ready-made features that a linear probe over the raw
embedding would otherwise have to relearn from a few thousand frames. On the
pre-revision 11-fold data it was +0.016, 7/11 folds up.

Run alone against `cv_baseline`, per IDEAS.md — composing with
`context-embedder` is a separate question and only worth asking if this stands
up on its own.

Known caution carried from the archived note: the sigmoid block is ~10x smaller
in scale than the embedding block and nothing normalises the two. Input
standardization is a separate parked idea (with its own zero-variance-dim NaN
bug); this run deliberately does not touch it, so it changes one thing.

## Changes

- Rebuilt `embedders/yamnet_combined/yamnet_combined.keras`, which was missing
  from disk (the `weights_keras2/yamnet.h5` its `BUILD.py` reads is gone from
  this checkout; recovered from `~/projects/buzzdetect-agentic`). Verified the
  1024-d embedding block is **byte-identical** (max abs diff 0.0) to the shipped
  `embedders/yamnet/yamnet.keras` on the same input, and the 521-d sigmoid block
  is in [0, 1]. Built straight into the shared `embedders/` tree — a new file
  under a name nothing else claims.
- No code changes. The embedder class was already in main.

## Results

| fold | cv_baseline | this exp | delta | buzz frames | val frames |
|---|---|---|---|---|---|
| Lily Adam - One Hive/willard/1_11 | 0.180 | 0.230 | **+0.050** | 305 | 6930 |
| JamesU - MustardBumbler/1_29 | 0.426 | 0.454 | **+0.028** | 2144 | 7617 |
| Luke - Diel Drivers/2026-04-08/1_150 | 0.021 | 0.014 | -0.007 | 146 | 4947 |
| Luke - Diel Drivers/2026-05-06/1_95 | 0.037 | 0.029 | -0.008 | 433 | 7572 |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 0.425 | 0.415 | -0.010 | 1031 | 7540 |

- mean sens@fpr0.005: baseline **0.218** -> this **0.228** (**+0.010**)
- 2 folds up, 3 down. Every fold reached the target FPR.

**Inconclusive.** +0.010 is well inside the repeat-run spread `harmonic-comb`
measured on this exact roster in the run just before this one: two CVs of one identical config
came out 0.016 apart, 3 up / 1 down / 1 flat. This result is smaller than that
and the folds are split, which is LOOP.md's definition of a result not worth
acting on. It is also smaller than the +0.016 the archive recorded at 7/11
folds, so nothing here contradicts the old direction either — it just cannot be
separated from noise at 5 folds.

The two movers do not line up with a story. willard (+0.050) is the thinnest
non-degenerate fold at 305 buzz frames, exactly where `eval-sampling-floor`
puts the sampling SD highest; mustard (+0.028) is the richest at 2144, where
the measured repeat-run movement was 0.019. One is up by about its own noise,
the other by a bit more than its own noise, and the three down folds are all
within 0.010.

**The one clean structural observation: it converges faster for the same
val_loss.** Best epoch fell on every fold that trains long — mustard 162 -> 129,
Fit+Fast 174 -> 130, 1_95 160 -> 156 — while `best_val_loss` is identical to
four decimals on the two big folds (0.9344 / 0.9344, 0.9294 / 0.9285). The 521
sigmoid scores are reaching the same optimum sooner, which is what you would
expect from features that are already a supervised readout of the block beside
them. Same destination, shorter path — not more information.

### Block scale, measured (for the normalization follow-up)

Over a sample of extracted frames, embedding block vs sigmoid block:

| | mean per-dim sd | mean abs value | max |
|---|---|---|---|
| embedding (1024-d) | 0.089 | 0.053 | 7.3 |
| sigmoid (521-d) | 0.011 | 0.004 | 1.0 |

So **~8x in per-dim spread and ~13x in magnitude** — the archived note's "~10x,
nothing normalizing the two" is confirmed on live data. With one shared Adam
learning rate and no per-block scaling, the sigmoid dims get a proportionally
smaller share of the gradient than their information content warrants. That is
the standardization idea in IDEAS.md, and this run deliberately did not touch it
(one change at a time).

Also confirmed live: the **zero-variance-dim hazard** that idea flags. In the
sample, ~120 of the 521 sigmoid dims have per-dim sd < 1e-4 (AudioSet classes
that never fire on this corpus at all). A Keras `Normalization` layer divides by
`sqrt(var + 1e-7)`, which on those dims is a ~1e4x amplification of nothing —
the NaN blowup the archive hit. Mask, floor, or drop them before adapting.

## Conclusion

`yamnet-combined` does **not** revalidate on the post-2026-09-08 data as a
result worth building on: +0.010 headline, 2 folds up / 3 down, against a
measured repeat-run spread of 0.016 on this roster. It is not a negative
either — the direction matches the archive and nothing here rules out a real
+0.01-ish effect; 5 folds simply cannot resolve one. Contrast
`context-embedder`, which on the same data and the same day moved +0.040 with
the three measurable folds all up: that is what a result that survives this
roster looks like.

Do not compose it with `context-embedder`. The precondition IDEAS.md set —
"only then ask whether it composes" — was that it stand up on its own, and it
did not.

What is worth carrying forward is the diagnostic, not the delta: the two blocks
really are ~10x apart in scale, and the concatenation as it stands is asking one
learning rate to serve both. If the sigmoid scores are ever going to add
anything, standardization is the precondition, not an optimization on top of a
win. The archived measurement of that (+0.014 on top of combined, 6/8 folds)
was never adopted only because the LR and patience were then tuned for the old
input scale — which is a fixable problem, not a verdict.

### Artifacts

- `embedders/yamnet_combined/yamnet_combined.keras` is now on disk in the shared
  tree (it was missing; rebuilt from the `yamnet.h5` in
  `~/projects/buzzdetect-agentic`). Anyone rerunning this or the standardization
  follow-up needs no re-extraction.
- `02_set/sets/medium/embeddings/yamnet_combined/` — 506 MB, 6.8 min at
  `--workers 2` on CPU. Shared cache, survives this worktree.
- CV: ~13 min for 5 folds. Model at
  `.local/worktrees/combined-revalidate/models/combined_revalidate/`.
