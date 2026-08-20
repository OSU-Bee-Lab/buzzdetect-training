# context-embedder

## Hypothesis

`exp/context-stack` widened each frame with its neighbours (t-1, t, t+1) and gained
+0.050 on `sens_persite` — the largest gain in the CV log. It did that inside
`03_train`, which left two defects:

- **It can't ship.** The generated `model.py` feeds 1024-d embeddings to a model
  that wants 3072.
- **Its eval is optimistic.** It stacked cached embeddings, and the cache groups
  frames by label, so a frame's "neighbours" were other frames carrying its own
  label. Continuous audio doesn't work that way.

Moving the stacking into a new **embedder** fixes both: the context frames come
from the audio that actually sat either side of the frame in the snip, and
inference gets the wider representation for free because the embedder is part of
the shipped path.

Prediction: a gain over the 0.206 baseline, smaller than context-stack's 0.256 —
that number is the optimistic ceiling, and what survives here is the honest one.

## Changes

### `embedders/yamnet_context/embedder.py` (new)

Subclasses `EmbedderYamnet`. `embed(audio)` takes a buffer of **contiguous**
audio, runs YAMNet over it, and returns `concat(frame t-1, frame t, frame t+1)`
per frame. `n_embeddings = 3072`; `context_frames = 1` is the knob.

Memory chunking (`cfg.CHUNK_FRAMES`) happens **inside** `embed()`, on frame
boundaries, so no caller can split the buffer and silently clamp a frame's
context. Frames at the two ends of a buffer repeat themselves (edge clamping) —
2 frames per buffer, and the same thing an inference buffer's edge does.

Nothing in `03_train` changed: the probe's input layer is sized from
`n_embeddings`.

### `02_set/extract.py` — the part the hypothesis actually depends on

**This was not optional, and it is the finding that shaped the experiment.**
Extraction does not hand the embedder contiguous audio. `extract_ident_both`
frames each coverage chunk in time order, then buckets the frames by their
collapsed label string (`ins_buzz_high+mech_plane_prop.pickle`), concatenates
each bucket, and embeds *that*. Every frame in a bucket carries the same label
by construction, so an embedder stacking within the buffer it is handed would
reproduce context-stack's optimistic geometry exactly — plus fresh
discontinuities at every `CHUNK_FRAMES` split. The experiment would have
measured nothing.

So the embedding step was moved ahead of the bucketing, gated on
`context_frames`:

- Each coverage chunk's frames are embedded in one `embed()` call, in time
  order; the resulting embeddings are bucketed by label instead of the audio.
- `extract_ident_embeddings` (the cached-audio-to-embeddings path) raises for a
  context embedder — the cache is bucketed, so context cannot be recovered from
  it. Idents whose audio is already cached are rebuilt from the snips instead,
  with `write_audio=False` so the audio cache — shared with every embedder at
  this sample rate and frame length — is read but never rewritten.
- A rescued frame (cut off the grid to save a short event) has no neighbours to
  hand it; its context clamps to itself.

With `context_frames = 0` — every existing embedder — this path is unchanged.

### Chunk boundaries

The stated risk was `CHUNK_FRAMES`. Three boundaries actually exist, and here is
what each got:

| boundary | handling |
|---|---|
| `CHUNK_FRAMES` memory split | **Removed.** Chunking moved inside `embed()`, on frame boundaries, over one contiguous buffer. No frame loses a neighbour to it. |
| label bucket | **Removed.** Embedding now happens before bucketing, so it is not a boundary at all. |
| coverage chunk / snip edge | **Clamped** (first and last frame repeat themselves). Two frames per chunk, at a real edge of the available audio; the same clamp inference does at a buffer edge. |


### `02_set/extract.py` — one more guard added this session

`WorkerExtract.__init__` now raises if `context_frames` is set and
`config_extract.framehop_prop != 1`. The contiguous buffer handed to `embed()`
is `np.concatenate(frames)`, which only reconstructs the chunk's real audio when
frames tile it without overlap; at `framehop_prop < 1` it would splice
duplicated audio instead, and the existing frame-count check wouldn't catch it
(the count comes out right either way). `medium` extracts at `framehop_prop=1`,
so this doesn't change the run — it just closes a silent-corruption path for
whichever config runs next.

## Results

Extraction verified before training: 2625 ident dirs matching `yamnet`'s
coverage exactly, vectors 3072-d on disk, and a direct stacking check —
`out[i, 1024:2048] == out[i+1, 0:1024]` (frame *i*'s centre slot is frame
*i+1*'s left-context slot), both buffer edges clamped, and the centre slot of
`embed()` bit-identical to plain YAMNet's `embed_frames()`.

| fold | baseline sens@fpr0.005 | this exp | delta | val frames |
|---|---|---|---|---|
| Luke - Various Opportunistic Recordings/2025-08-05/31 | 0.157 | 0.297 | +0.140 | 942 |
| JamesU - MustardBumbler/1_29 | 0.448 | 0.550 | +0.102 | 6984 |
| Luke - Various Opportunistic Recordings/2025-07-03/1_37 | 0.279 | 0.336 | +0.057 | 4715 |
| Luke - Diel Drivers/2026-04-08/1_150 | 0.028 | 0.068 | +0.040 | 4947 |
| Luke - Diel Drivers/2026-05-06/1_95 | 0.027 | 0.042 | +0.015 | 6628 |
| Luke - Various Opportunistic Recordings/2025-08-27/48 | 0.013 | 0.023 | +0.010 | 1571 |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 0.386 | 0.390 | +0.004 | 4712 |
| Luke - Various Opportunistic Recordings/2025-06-23/1_23 | 0.326 | 0.314 | -0.012 | 315 |
| Lily Adam - One Hive/recorders/wooster/2024-07-26/1_143 | 0.260 | 0.247 | -0.013 | 4708 |
| Luke - Various Opportunistic Recordings/2025-08-12/1_114 | 0.163 | 0.122 | -0.041 | 3768 |
| Lily Adam - One Hive/recorders/willard/2024-08-07/1_11 | 0.177 | 0.118 | -0.059 | 4730 |

- sens_persite @ fpr0.005: baseline 0.206 → this 0.228 (+0.022)
- context-stack (optimistic ceiling, cached-embedding eval): 0.256

7 of 11 folds moved up, 4 down. The four losers are not small-buzz folds where a
swing is noise (`wooster` 4708 val frames, `1_114` 3768, `willard` 4730 —
mid-pack by `frames_val`; `1_23` at 315 frames is the one loser worth
discounting). The two biggest gains (`2025-08-05/31` +0.140, `MustardBumbler`
+0.102) are on folds with real buzz mass, not noise either.

`willard` (-0.059) is the same fold IDEAS.md's `willard-regression` flagged for
`exp/context-stack` (-0.074 there). Temporal context hurts that deployment
specifically, in both the optimistic and the honest version — evidence it's a
property of the fold, not an artifact of context-stack's same-label neighbour
bug. `willard-regression`'s proposed diagnostic (listen to the frames where the
two models disagree most) is now more interesting to run, since the honest
embedder reproduces the regression.

## Conclusion

Confirmed: folding real temporal context into the embedder — not the cached,
same-label-neighbour version — still gains on `sens_persite` (+0.022, 0.206 →
0.228), with the direction holding on 7 of 11 folds including the two largest.
It ships, unlike `context-stack`: `model.py`'s input layer sizes from
`n_embeddings` (3072) automatically, and inference calls the same `embed()`
extraction used, so no separate stacking step is needed at serving time.

The honest number is well below context-stack's optimistic 0.256, as expected —
same-label neighbours in the cached-embedding version were doing real (if
illegitimate) work. The gap between 0.228 and 0.256 is roughly the size of that
artifact.

One fold (`willard`) loses under both versions of temporal context, which
argues the loss there is a real property of that deployment rather than noise
from the cached-eval bug. Worth a `willard-regression`-style listening pass
before trying a further context variant (e.g. wider `k`) on top of this.
