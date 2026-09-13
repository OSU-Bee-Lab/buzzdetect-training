# asymmetric-context — KILLED 2026-09-13 (deployability audit)

CV run killed at fold 3/8 (pid 598148, plain `kill`). Verdict: **not deployable
as built.**

## Why

`03_train/dataset.py::apply_context()` is a train-time transform over the
*extraction cache*: it opens `frametimes.csv` under each ident directory,
infers the frame grid from the modal gap between frame starts, and joins a
frame's temporal neighbour out of a **sibling label-pickle in the same ident**.

Nothing in the deployment path can do that. The shipped artifact is a single
ONNX graph (`tools/export_onnx.py` → `buzzdetect/engine/src/inference/models.py`):
waveform in, predictions out, "no embedder plugin, no NumPy front end, no
TensorFlow". It is handed a chunk of raw samples and has no cache, no
`frametimes.csv`, and no cross-pickle index. There is no hook to add one.

## Second, independent problem: train/deploy distribution mismatch

The run's own log lines show the neighbour-hit rate varies enormously with how
much of a recording happened to be annotated:

    train folds  ~99.3–99.7% real neighbours
    eval folds    62.6–94.8% real neighbours

At deployment, on contiguous audio, the clamp rate is ~0 (two frames per chunk).
And in `asymmetric` mode a clamped frame yields `centre - centre = 0` for the
whole difference block — a distinctive input pattern that appears in 5–37% of
eval frames here and essentially never in the field. So even a perfectly
faithful deployable reimplementation would not see the inputs this model was
fitted on.

## The hypothesis itself is salvageable — the mechanism is not

`[centre, centre - mean(neighbours)]` is a fixed linear map of the concat-context
vector, so it can live entirely inside an embedder: subclass
`embedders/yamnet_context/` and override `stack_context()`. That is the only
form worth running. **But see the pre-existing extraction bug below first** —
today it would be trained on the wrong neighbours anyway.

## Pre-existing bug found while auditing this (affects `yamnet_context`)

`embedders/yamnet_context/embedder.py` docstring claims:

> "02_set/extract.py checks `context_frames` and does exactly that [hands
> embed() contiguous audio]; embedding a buffer of frames that were first
> grouped by label would make each frame's 'neighbours' other frames with its
> own label, which is the artifact this embedder exists to avoid."

`grep -rn context_frames 02_set/` returns **nothing**. `extract_ident_both()`
(02_set/extract.py ~line 845) builds `frames_by_label`, concatenates each
label's frames, and calls `self.embedder.embed(chunk)` on that. So
`yamnet_context` / `yamnet_context_aves` *do* get label-grouped buffers, and
their "neighbours" *are* other same-label frames from elsewhere in the
recording — exactly the artifact the docstring says is avoided. Every logged
`*-context` result is affected. **Needs Luke's call**; not fixed here.
