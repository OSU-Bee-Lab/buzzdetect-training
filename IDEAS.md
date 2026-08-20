# Experiment Ideas

Check this file and `log.jsonl` before proposing an experiment, then take it
through the lifecycle in `LOOP.md`.

`log.jsonl` holds CV-era runs only — measured by leave-one-fold-out on day-long
annotated recordings, against `sens_persite`. Those numbers are comparable to
each other and are the ones to beat.

Everything under [Tried before the rework](#tried-before-the-rework) comes from
an era with a different metric, a different eval corpus, and a training set that
was a scattershot of collection methods. Its numbers are gone and its verdicts
are unreliable — see the warning at the head of that section before you treat
any of it as settled.

---

## Open ideas

### Different levels of automatic annotation generation
From bee hive and from nighttime audio

### context-embedder

**Hypothesis:** `exp/context-stack` showed that widening each frame with its
neighbours (t-1, t, t+1) is worth +0.050 — buzz is sustained, and the hard
negatives are transient. But it was implemented in the training code, which
leaves two problems: the shipped model can't run (inference feeds 1024-dim
embeddings to a 3072-dim model), and every frame in a snip shares that snip's
label, so the neighbours always agree in a way continuous audio won't reproduce.

**What to do:** implement the same idea as a *new embedder* instead — a wrapper
over YAMNet that emits the stacked representation. Then:

1. Extract embeddings for the set under it. The snips carry a generous buffer of
   audio before and after the labeled events, so the context frames are real
   neighbouring audio rather than repeated edges or same-label padding.
2. Train unchanged — the probe just sees a wider embedding.
3. Inference gets it for free, because the embedder is part of the shipped path.

**Why it might help:** it is the same mechanism that already worked, measured
honestly and in a form that can actually ship. The +0.050 is an upper bound;
what survives the honest eval is the real number.

**Caveats:** costs a re-extraction. Sweep k after the k=1 result is confirmed —
if duration is the mechanism, k=2 or 3 should keep helping until the window
starts smearing onsets.

### input-standardization

**Hypothesis:** `exp/yamnet-combined` (+0.016) concatenated YAMNet's 521 sigmoid
class scores onto the 1024-d embedding, and the two blocks are on very different
scales — score block mean 0.004 / max 0.41 against embedding mean 0.041 / max
2.6, roughly 10x. Nothing in `03_train` normalizes its input. `Dropout(0.2)`
drops units uniformly regardless of scale, so the small block is effectively
regularized far harder than the large one, and the probe may never get to use
the scores properly.

**What to do:** standardize per feature block (or per dimension) before the
probe. Cheapest honest version is a fixed transform computed on the training
folds only and applied at scoring, not a `BatchNormalization` layer —
BatchNorm was tried in the fixed-test era and was a clear negative because its
running statistics didn't transfer across deployments, which is a real risk here
too.

**Why it might help:** it decides whether `yamnet-combined`'s modest gain is the
ceiling or an artifact of the scores being drowned out. Answer that before
composing combined with anything else.

**Caveats:** no re-extraction needed — the `yamnet_combined` embeddings for
`medium` are already cached. Applies equally to any future concatenated
representation, `context-embedder` included.

### willard-regression

**Hypothesis:** `exp/context-stack` gained in 8 of 11 deployments but lost 0.074
at `Lily Adam - One Hive/recorders/willard/2024-08-07/1_11`, a large fold. There
is something specific about that deployment that temporal context hurts.

**What to do:** pull the frames where the two models most disagree at their
per-fold thresholds and listen. Not a training experiment — a diagnostic that
should precede the next context experiment.

### near-chance-deployments

**Hypothesis:** two deployments (`Luke - Diel Drivers/2026-05-06/1_95` and
`Luke - Various Opportunistic Recordings/2025-08-27/48`) sit near zero for every
model tried, including one with 430 buzz frames — so it is not a small-sample
artifact. Whatever is wrong there is the largest single source of headroom in
the endpoint, since the endpoint averages deployments equally.

**What to do:** diagnostic first. Are the buzzes audible? Is the annotation
right? Is the recorder gain or placement different? Answer that before designing
a training change around it.

### aves-intermediate-layer

**Hypothesis:** AVES embeddings from the last transformer layer are too
bird-specific to discriminate insect buzz; an intermediate layer (6-9 of 12)
carries more general acoustic features.

**What to do:** in `embedders/aves/embedder.py`, change `layer_outputs[-1]` to
`layer_outputs[N]` for N in {5, 7, 9}; re-extract; train. `n_embeddings` stays
768, so no other code changes.

**Background:** `aves_lite` performed at chance. AVES embeddings are symmetric
around zero for both classes, where YAMNet's are ReLU-sparse and linearly
separable. Wav2vec2 transfer literature consistently favours middle layers.

**Caveats:** LOOP.md currently constrains the embedder to YAMNet, so this needs
that constraint lifted first. Requires a re-extraction per layer tried.

### lora-adapter

**Hypothesis:** LoRA adapters inside the frozen backbone, training only adapters
plus head, improve embedding quality with far less overfitting risk than full
fine-tuning.

**Caveats:** LoRA is a transformer technique; inserting it into YAMNet's conv
layers is non-standard. Best first target is a transformer embedder. Note that
straightforward backbone fine-tuning already failed badly on the old data
(`yamnet-ft`, -8.3pp) — that verdict is from the bad-data era and may not hold,
but the overfitting mechanism it described is plausible either way.

---

## Tried before the rework

**Read these as leads, not verdicts.** They were measured on a fixed
train/validate split against a retired hand-curated corpus, with a training set
of mixed provenance, at a time when single-run variance was wide enough
(~0.16-0.25 on the same config) that several entries were later invalidated as
dataset artifacts.

The clearest reason not to trust them: **`temporal-context` — concatenating
[prev, curr, next] frames — was logged as a clear negative (-2pp). The identical
change, rerun as `exp/context-stack` on the current set and metric, is the
largest gain yet (+0.050).** A verdict inverted. Assume any of the below could
do the same, and rerun rather than defer to it.

Full entries: `.local/archive/log_precv.jsonl`. Working trees and notes:
`.local/worktrees-fixed-test/`.

One has already been retested under CV: the fixed-test entry `combined-embedder`
was about validation scope, but the `yamnet_combined` *embedder* it left behind
was finally extracted and trained as `exp/yamnet-combined` (+0.016).

| Area | What was tried | Old verdict |
|---|---|---|
| Regularization | Dropout(0.2) + label smoothing 0.2 | The whole gain over an unregularized probe (3.6pp). Current default. |
| | Label smoothing 0.3 | Collapses. 0.2 was the peak of a monotone trend. |
| | L2(1e-4), alone or added | Indistinguishable from no regularization. |
| | BatchNorm on input embeddings | Clear negative (-5.8pp) — training-set running stats didn't transfer. |
| Head shape | Dense(128, relu) before output | Worse than a linear probe, replicated twice, no overfitting signature. |
| Input surgery | Bandpass 100-3000 Hz before YAMNet | Clear negative (-4.7pp); corrupts YAMNet's expected input. |
| | Zeroing mel bins above 3000 Hz | Catastrophic (-14.3pp). |
| | Handcrafted frequency features | Neutral twice; YAMNet already encodes it. |
| | White-noise samples as 'static' | Neutral; the false positives are structured, not broadband. |
| Backbone | Fine-tune YAMNet layers 13-14 at 1e-5 | Clear negative (-8.3pp); overfit, train 78% vs val 59%. |
| Class weighting | 2x buzz upweight over balanced | Negative-to-neutral; balanced weights already fine. |
| Loss | Focal loss, alpha 0.25 and 0.75 | Shifts the operating point, doesn't lift the curve. |
| Translation | Binary (all non-buzz collapsed) | Hurt; multi-class auxiliary supervision helped. Retested under CV — see `binary-translation-cv`, neutral on the endpoint. |
| Training procedure | min_delta=0.002 early stopping | 6.6x variance reduction, no mean change. Adopted as default. |
| | Forcing buzz out of validation | Negative; early stopping needs buzz in the monitor fold. |
| Temporal | [prev, curr, next] frame concatenation | Negative (-2pp) — **and now known to be wrong**, see above. |
