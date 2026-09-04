# Experiment Ideas

Check this file and `log.jsonl` before proposing an experiment, then take it
through the lifecycle in `LOOP.md`.

`log.jsonl` holds CV-era runs only — measured by leave-one-fold-out on day-long
annotated recordings, against mean per-fold sens@fpr0.005. Those numbers are comparable to
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

### stopping-rule-scale

**Hypothesis:** `EarlyStopping`'s `min_delta=0.002` on `val_loss`
(`03_train/train.py:257`) was tuned for the 1024-d linear probe at
`framehop_prop=1`. It is an absolute threshold on a loss whose scale and
per-epoch improvement both depend on input width and on how many frames an
epoch contains, so it is not obviously portable to any other structure — and
four runs on 2026-08-20/21 all stopped earlier than baseline, monotonically in
input width:

| config | width | median best_epoch |
|---|---|---|
| cv-baseline | 1024-d | 56 |
| context-embedder (k=1) | 3072-d | 38 |
| context-width (k=2) | 5120-d | 32 |
| framehop-overlap (2x frames/epoch) | 1024-d | 13 |
| context-pooling (attention) | 3072-d | 4 |

If this is real, every structural experiment in the log has been partly
measuring a stopping rule rather than a representation, and the two context
results in particular (`context-embedder` +0.022, `context-width` -0.043 vs k=1)
are confounded with it.

**What to do:** one CV at k=1 with a smaller `min_delta` (or a relative
criterion), against `exp/context-embedder` as the paired comparator. If k=1's
number moves materially, rerun `context-width` before believing the
dose-response turnover.

**Caveats:** `framehop-overlap`'s agent found per-fold epoch ratio correlates
only weakly with per-fold delta (r=0.155), so early stopping does not explain
*which* folds moved there — the effect may be real but not the whole story.
Also note `min_delta=0.002` was itself adopted for a reason (6.6x variance
reduction pre-rework), so loosening it may trade bias for variance.

### noise-floor-cv

**Hypothesis:** run-to-run variance on this pipeline is far larger than the loop
assumes, and single-run per-fold deltas carry little signal.

**Why now:** `exp/tail-loss-retest` ran a `weight=0.0` control — mathematically
identical to `cv-baseline`'s loss — on `JamesU - MustardBumbler/1_29` and got
sensitivity 0.002-0.070 against the logged 0.448, then reproduced the same
collapse through main's completely unmodified `train.py`. The driver
(`tools/diag_tailloss.py` on that branch) calls the same `_load_data` /
`_train_one` / `_write_scores` with the same fold composition, `epochs=400`,
`patience=50` as `train_set`'s rotation loop, and `frames_train` matches
baseline's to within ~100 frames — so no harness difference has been found to
explain it.

**What to do:** the control LOOP.md names as expensive and never run — repeat the
baseline config as a full CV under a fresh `--name`, join per fold against
`models/yamnet_medium_general`, and report the per-fold spread. That number is
the denominator for every delta in this log.

**Caveats:** interacts with `stopping-rule-scale` — if stopping is unstable,
some of this "seed noise" may be stopping-rule noise with a fixable cause.

### event-level-metric

**Problem, not yet a hypothesis:** `predictions.csv` stores only
`activation_ins_buzz` and `correct` per frame (`03_train/train.py:148-151`) —
no event id, no timestamp. So there is no way to ask "did any frame overlapping
this annotated event clear the threshold?", and every comparison is locked to a
per-frame rate.

That made `exp/framehop-overlap` uninterpretable: halving the hop doubled both
the buzz frames and the negative population the FPR threshold rests on
(`buzz_frames_total` 5641 -> 11225, `neg_frames_fold_median` 22 -> 43), so its
-0.152 is measured over a different frame population than baseline's and cannot
be read as a capability gap in either direction.

**What to do:** carry event identity (or frame start time) into `predictions.csv`
and add an event-level read alongside the frame-level one. Any future experiment
that changes frame density — hop, framelength, rescue policy — needs it to be
comparable at all. Note this changes what a model directory contains, not the
metric itself, so it does not run afoul of the `03_train/metrics.py` rule.

### normalization-zero-variance

**Status:** the `standardization-convergence` sweep this replaces was run as
`exp/std-convergence` and came back negative-to-inconclusive — see `log.jsonl`.
What it left behind is a concrete bug rather than an open hyperparameter
question.

**The problem:** 52 of `yamnet_combined`'s 1545 input dims have ~zero variance
across the training folds. A `Normalization` layer divides by
`sqrt(var + 1e-7) ~ 3e-4` on those dims, amplifying anything nonzero by ~3000x.
This produced a genuine NaN blowup at lr=5e-4 without gradient clipping, and it
is structural — no learning rate avoids it.

**What to do if anyone revisits standardization:** mask or floor the
near-zero-variance dims before adapting the layer (a variance floor well above
1e-7, or drop those dims outright — they carry no training signal by
definition). Until that's done, any standardized-input result is running on a
numerically fragile setup and its failures can't be attributed to the
hypothesis.

### willard-regression

**Hypothesis:** `exp/context-stack` gained in 8 of 11 deployments but lost 0.074
at `Lily Adam - One Hive/recorders/willard/2024-08-07/1_11`, a large fold. There
is something specific about that deployment that temporal context hurts.

**Strengthened by `exp/context-width`:** the regression scales with context
width, monotonically — baseline 0.177, k=1 0.118 (-0.059), k=2 0.066 (-0.111).
Roughly double the hit for double the window, on a fold that is mid-pack in
`frames_val` (4730), so this is not a thin-fold swing. `exp/deployment-forensics`
attributes it to willard having the highest fraction of short (<1s: 68%) and
isolated (>5s gap: 68%) buzz events of any fold — stacking dilutes a brief
isolated buzz with silent neighbours, and a wider window dilutes it further.

**What to do:** pull the frames where the two models most disagree at their
per-fold thresholds and listen. Not a training experiment — a diagnostic that
should precede the next context experiment. If the dilution story survives
listening, the follow-up is context width that adapts to event length rather
than a single fixed k.

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
layers is non-standard. Best first target is a transformer embedder.

**Update (2026-09-01):** straightforward differential-rate fine-tuning of
YAMNet layers 13-14 (`exp/trunk-ft`, `trunk-ft-1e5`) is **+0.046 vs a matched
frozen control** under CV — the pre-rework `yamnet-ft` -8.3pp verdict inverted,
like `temporal-context` did. So the "backbone fine-tuning overfits" premise no
longer holds on the current set, and LoRA's motivation (fine-tune with less
overfit risk) is weaker than it was — though a cleaner low-rank adapter could
still beat plain FT. See `exp/trunk-ft` notes. Open follow-ups there: LR/patience
sweep tuned for the batch-1024 regime and augmentation on top of the
fine-tune.

**Depth is settled on `medium` (2026-09-04):** `exp/unfreeze-more` took the
unfreeze one block deeper (layers 12-14) and the dose-response turns over —
frozen 0.216 → 13-14 **0.262** → 12-14 0.229, with median `best_epoch`
falling 35 → 25. 13-14 is the operating point; deeper only overfits. Still
open on the denser `large` set, where the extra capacity might be fed.

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
