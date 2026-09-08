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

## Conclusion
