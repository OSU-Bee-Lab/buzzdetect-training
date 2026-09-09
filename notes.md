# aves-probe

## Hypothesis

AVES (`aves-base-bio`, wav2vec2 pretrained on bioacoustics) as a drop-in
replacement for YAMNet's frozen 1024-d embedding, final transformer layer,
768-d. Everything else is `cv-baseline`'s config: linear probe, Dropout(0.2),
label smoothing 0.2, Adam 0.002, `medium`, `--translation general`.

This is IDEA E1's prerequisite, not E1 itself. E1 proposes a *middle*-layer
sweep ({5,7,9}); this run establishes whether the last layer is anywhere near
competitive on the current data and the current metric before spending a
re-extraction per layer.

Why it is worth a run at all: the only AVES result on record ("performs at
chance") is from 2026-06-04, two eras back — measured on `aves_lite`, scored on
the **retired** stage-4 precision endpoint, never logged, no surviving model
dir. Two data revisions have landed since, and `temporal-context` ->
`context-stack` is the standing example of an inherited negative inverting.

The comparison is unusually clean for an embedder swap: AVES is 1.0 s / 16 kHz
against YAMNet's 0.96 s / 16 kHz, so at `framehop_prop: 1` the frame grid is
near-identical and `overlap_event_s` moves 0.192 -> 0.200. That is a 4% shift
in the labelling rule, against the 5.2x shift that made `perch-probe`
uninterpretable. Its headline should join `cv_baseline` directly.

Expectation: a loss. The surviving diagnostic from 2026-06 is a property of the
embeddings rather than of the metric — AVES activations are symmetric around
zero for both classes with roughly half YAMNet's dimension-wise buzz/non-buzz
separation (0.13 vs 0.23), where YAMNet's are ReLU-sparse, non-negative and
linearly separable. A linear probe is the worst head for that geometry. A clear
negative here closes AVES-as-drop-in on current data; a near-miss makes the
middle-layer sweep worth its extractions.

## Changes

None to the *experiment* — the probe, the set, the folds and the translation are
`cv-baseline`'s. Extraction is under a new `--embedder aves` name, so it writes
where nothing else claims and the symlinked shared cache stays intact.

Two things had to be done to run it at all, both committed to **main** rather
than this branch, because both are shared infrastructure that every future AVES
run wants:

1. The gitignored 377 MB `aves-base-bio.pt` checkpoint was missing; fetched with
   `embedders/aves/BUILD.py`.
2. `embedders/aves/embedder.py` ran one 1.0 s frame per forward pass on the CPU
   — 0.19 s/frame, **14.3 h** for `medium`'s 267,742 frames. Rewritten to batch
   and run on the GPU: **1.0 h** at batch 64, 1.25 GB peak VRAM on the 4 GB
   GTX 1650 (main, `8fe1336`).

   |          | s/frame | full set |
   |---|---|---|
   | CPU bs=1 (original) | 0.1923 | 14.3 h |
   | CPU bs=64 | 0.1489 | 11.1 h |
   | CUDA bs=64 | 0.0139 | 1.0 h |

   Throughput is flat from batch 16, so the GPU is saturated by a single stream.
   Output is unchanged against the old path — max abs diff 6.7e-6 on a scale of
   0.85, per-frame cosine >= 0.9999998, i.e. float associativity from batched
   GEMMs — and no `aves` embeddings existed on disk to be made inconsistent.

   Two hazards found on the way, both recorded in `HANDOFF.md`: cuDNN in the
   pinned env cannot finalize this model's conv1d descriptors
   (`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`, worked around with torch's
   native kernels rather than by repinning anything), and the extraction must
   run `--workers 0`, because `main.py` imports TensorFlow first, TF initializes
   the CUDA driver in the parent, and `fork` then poisons the child's context.

**The "AVES is horrifically slow" folklore was about the implementation, not
the model.** One unbatched CPU forward pass per frame is 14 h for this set;
batched on the GPU the same embeddings take 1 h. Not benchmarked against
YAMNet's own per-frame cost here, so no claim is made about the ratio — but at
~1 h per full extraction, the E1 middle-layer sweep is no longer gated on
extraction cost the way the parked note assumed.

## Results

**Clear negative. 0.218 -> 0.074 (-0.144), 5 folds down, 0 up.**

| fold | buzz frames (base/aves) | base | aves | delta |
|---|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 2144 / 2046 | 0.426 | 0.016 | **-0.410** |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 1031 / 991 | 0.425 | 0.205 | **-0.220** |
| Lily Adam - One Hive/.../1_11 | 305 / 296 | 0.180 | 0.120 | -0.060 |
| Luke - Diel Drivers/2026-05-06/1_95 | 433 / 408 | 0.037 | 0.017 | -0.020 |
| Luke - Diel Drivers/2026-04-08/1_150 | 146 / 144 | 0.021 | 0.014 | -0.007 |

The two folds worth reading carry the effect, and they are the two largest
losses. Nothing here is at the noise floor (~0.016); `1_29` alone moves 26x it.

**The 4% labelling shift is not the explanation.** `overlap_event_s`
0.192 -> 0.200 cost 174 buzz frames out of 4059 (-4.3%) against 1403 fewer
frames overall (-4.1%) — the buzz *rate* per fold is unchanged to three
decimals (`1_29`: 0.2815 -> 0.2801). This is the dilution check that made
`perch-probe` uninterpretable, and AVES passes it. The headline join to
`cv_baseline` is sound.

Training behaviour matches: `1_29` peaks at `val_sens 0.0020` in epoch 3 and
decays to 0 — the probe never finds a usable direction on the richest fold.

### The 2026-06 geometry diagnostic does not reproduce; it inverts

Recomputed on the current `medium` embeddings, both embedders, the 5 rotating
folds, `ins_buzz` vs rest (34,606 YAMNet / 33,203 AVES frames, 11.7% buzz in
both):

| | mean \|Cohen d\| | mean \|AUC-0.5\| | top-10 dims \|AUC-0.5\| | frac exact zeros |
|---|---|---|---|---|
| YAMNet 1024-d | 0.155 | 0.022 | 0.260 | 0.896 |
| AVES 768-d | **0.242** | **0.074** | 0.226 | 0.000 |

The parked claim was AVES at 0.13 against YAMNet's 0.23, i.e. *half* the
dimension-wise separation. On current data the **average** AVES dimension
separates buzz better than the average YAMNet one, on both a parametric and a
rank measure. The 2026-06 number looks like a scale artifact: YAMNet's
activations are 89.6% exact zeros, so a mean taken across its basis is diluted
by ~900 dead dimensions, while AVES's are dense and zero-centred (50.4%
negative, overall mean -0.0003). Compare where the signal actually lives and
YAMNet wins narrowly (top-10 dims 0.260 vs 0.226, max \|d\| 1.91 vs 0.91) — its
information is concentrated in a few sparse, individually-strong,
ReLU-non-negative directions, which is exactly what a linear probe wants.

So the surviving half of the old diagnostic is the *shape* claim (sparse and
concentrated vs dense and distributed), not the *magnitude* claim. Retire the
0.13-vs-0.23 number rather than re-citing it.

## Conclusion

**AVES's final layer is not a drop-in for YAMNet on current data, and the
question is settled rather than inherited.** The 2026-06 "performs at chance"
verdict was measured on `aves_lite` under the retired stage-4 endpoint and was
worth no weight; this is the same conclusion re-measured on the current roster,
the current annotations and the current metric, with a clean frame join and a
dilution check that passes. 0.074 puts the last layer barely above the two
near-chance folds' own baseline sensitivity.

**Closing E1 (AVES intermediate layers) anyway, but not on the argument the
handoff expected.** The cheap pre-filter it asked for was run and came back the
wrong way — the geometry does *not* argue against middle layers, so E1 cannot be
closed on that. It is closed on the CV magnitude instead: -0.144 with both rich
folds collapsing is not the near-miss that would make a per-layer re-extraction
worth its cost. A middle layer would have to recover ~0.15 sensitivity that the
final layer does not have anywhere in it, and nothing measured here suggests
where that would come from.

What would reopen it is a *different* head, not a different layer. The one real
finding is that AVES's buzz information is distributed across many
weakly-separating dimensions where YAMNet's is concentrated in a few strong
ones, and a linear probe over Dropout(0.2) is the worst available reader of the
former. That is a head experiment on embeddings that now exist on disk, not an
extraction sweep — but it is blocked by the standing 1-layer injunction, so it
is parked rather than proposed.

**Infrastructure banked regardless of the verdict:** AVES extraction went
14.3 h -> 1.0 h (main, `8fe1336`) and `medium`'s AVES embeddings are in the
shared cache. Any future AVES question is now cheap to ask.
