# context-frames-fix

## Hypothesis

`embedders/yamnet_context/embedder.py` and `embedders/yamnet_context_aves/embedder.py`
both document, in their class docstrings, that `02_set/extract.py` checks
`context_frames` and hands them **contiguous real audio** padded either side.
It does not. `grep -rn context_frames 02_set/*.py` on `main` returns nothing:
the padded-context extraction was written on `exp/yamnet-aves-context` in the
previous era and never merged, while the embedders' docstrings were.

So on `main` every embedder, context or not, gets the default path:
`extract_ident_both` buckets frames **by collapsed label**, concatenates each
bucket, and embeds the concatenation. A `yamnet_context` frame's "temporal
neighbours" are therefore whichever frames happen to sit beside it *in its own
label's pickle* — same label by construction, and frequently from a different
part of the recording or a different snip entirely.

That is `context-stack`'s leak (`artifact`, +0.050 → +0.022 honest), and it is
the exact failure `IDEAS.md`'s standing fact warns about: *"Reading neighbouring
audio at extraction time is honest; reading neighbouring cache rows is not."*

**Measured on the shipped `main` cache** (`02_set/sets/medium/embeddings/yamnet_context`,
the one `context-verify` trained on):

- `left_block(row i) == centre_block(row i-1)` for **98.4%** of rows within a
  label pickle — i.e. context is in-pickle adjacency, as predicted.
- Of those in-pickle adjacencies, **84.1%** are truly contiguous audio
  (`start[i] - start[i-1] == 0.96 s`) over all frames, but only **65.8%** for
  `ins_buzz*` labels. So **~34% of buzz frames** carry context spliced from
  elsewhere in the recording (observed gaps up to 3597 s) — and the splice is
  always another buzz frame.

The rate is label-correlated, which is the definition of a leak rather than
just noise.

*Prediction / falsifier:* if `context-verify`'s +0.085 was carried by the leak,
the honest re-extraction shrinks it materially — toward `context-embedder`'s
honest E3 +0.022. If it survives at comparable size, the context lever is real
and the bug was cosmetic.

## Changes

`02_set/extract.py` only. Ported from `exp/yamnet-aves-context` (commit
`7676e64`) onto current `main`, keeping main's later fixes to that file (the
`.part` format argument and the post-run extraction verification), which the
old branch predates:

- `WorkerExtract.__init__` reads `self.context_frames = getattr(embedder,
  'context_frames', 0)` and refuses `framehop_prop != 1` for a context embedder
  (a contiguous buffer cannot be rebuilt from overlapping frames).
- `_read_pad_frame()` / `_embed_with_context()`: a chunk's frames are handed to
  `embed()` as ONE contiguous buffer, padded by `context_frames` frames of real
  audio each side read from the snip (snips carry `SNIP_BUFFER_S` = 30 s), and
  the pad rows are discarded. A kept row can only edge-clamp where the padding
  itself was unavailable — a true edge of the source recording, which is the one
  place a live deployment also has no neighbour.
- Rescue frames (cut off-grid, and by construction the frames carrying a
  labelled event) get the same padded treatment individually.
- `extract_ident_both` keeps `embeddings_by_label` parallel to
  `frames_by_label`, and the write path uses it instead of re-embedding the
  concatenated label bucket.
- `extract_ident_embeddings` (the rebuild-from-audio-cache path) raises for a
  context embedder, and `extract_ident` promotes `handle == 'embeddings'` to
  `'both'` for one — the framed-audio cache is keyed only by
  `(samplerate, framelength)` and is label-grouped, so it cannot supply
  neighbours.

**Every one of those lines is gated on `self.context_frames`**, which is 0 for
every embedder but `yamnet_context` / `yamnet_context_aves`.

## Results

(pending)

## Conclusion

(pending)
