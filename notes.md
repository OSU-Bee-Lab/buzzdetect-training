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

None to tracked code. `embedders/aves/embedder.py` was already on main and
unmodified; only the gitignored 377 MB `aves-base-bio.pt` checkpoint was
missing, fetched with `embedders/aves/BUILD.py`.

Extraction is under a new `--embedder aves` name, so it writes where nothing
else claims and the symlinked shared cache stays intact.

## Results

<!-- filled in when the run lands -->

## Conclusion

<!-- filled in when the run lands -->
