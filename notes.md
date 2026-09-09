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

<!-- filled in when the run lands -->

## Conclusion

<!-- filled in when the run lands -->
