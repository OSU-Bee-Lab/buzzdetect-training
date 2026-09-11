# yamnet-aves-context

## Hypothesis

Two independent representation levers have paid in this era, and they attack
different weaknesses:

- **temporal context** (`context-embedder`, +0.040 shipped / +0.058 xfold-pooled)
  — YAMNet sees 0.96 s and cannot tell a transient from a stationary texture.
- **frequency coverage** (`yamnet-aves`, +0.025 xfold, two draws, 4/5 folds up)
  — a second, differently-encoding backbone beside YAMNet.

They have never been run together; both `yamnet-aves` and
`yamnet-aves-head-fixed` name the composition as their top lead. If the two are
genuinely disjoint, the composition is additive.

**Falsifier:** if context on top of `yamnet_aves` gives materially less than the
+0.058 context gave on plain YAMNet, the two levers overlap — AVES's 1.0 s
receptive field over a wav2vec2 stack already supplies part of what the
neighbour frames supply.

## FINDING (independent of the experiment): the context cache clamps, and the clamp is label-correlated

The first attempt did the widening at **train** time over the existing
`yamnet_aves` cache, looking neighbours up by source time via `frametimes.csv`
— no re-extraction. Neighbours were honest (looked up by time, never by
adjacency inside a label pickle, which is the `context-stack` leak). But 7.9%
of frames had no extracted neighbour and clamped to themselves, and **the clamp
rate correlates with the label**:

| | n | clamped-left | clamped-right |
|---|---|---|---|
| negative | 69,486 | 0.070 | 0.070 |
| buzz | 9,750 | 0.144 | 0.144 |

Base rate of buzz is 12.3%; among left-clamped frames it is 22.4%. A clamped
frame's context block is **byte-identical** to its centre block — a
zero-difference signature any dense layer reads for free, and one a deployment
never produces. Luke's standard: the rotating folds must look exactly like what
a real deployment sees. That rules the approach out regardless of effect size.
Run killed at fold 2/5 (`1_29` was +0.104 over the matched control — exactly the
shape the artifact would produce, which is the point).

**The same defect is in the shipped `yamnet_context` cache**, measured directly
rather than inferred — a clamped row there has its left 1024-block byte-equal to
its centre block:

| `yamnet_context` cache | n | clamped-left | clamped-right |
|---|---|---|---|
| negative | 72,045 | 0.067 | 0.067 |
| buzz | 10,019 | 0.140 | 0.139 |

2.1x, essentially identical. Extraction hands `embed()` one
`np.concatenate(frames)` per annotation-cluster chunk and
`EmbedderYamnetContext` edge-clamps at the buffer ends, so every chunk boundary
is a clamp.

**Mechanism, which is why this is structural and not incidental:** an isolated
1 s buzz event expands to a 2.592 s chunk = 2 frames, *both* at a chunk edge.
Negatives come from chunks around long annotations (a 300 s `ins_trill` gives
~300 frames, 2 of them edges). Buzz events are short, so buzz frames sit at
chunk edges far more often. This falls out of the geometry and will be in every
context cache built this way.

Exposure: `context-embedder` (+0.040/+0.058), `context-monitor` (0.307) and
`context-monitor-r2` (0.312). Not yet a claim that those gains are false — the
clamp is coarse and `fpr 0.005` admits ~30 negative frames per fold, so it
cannot clear that bar alone, only re-rank within the top of a good score. It
has simply never been measured. The honest cache below is what measures it.

## Changes

**`embedders/yamnet_context_aves/`** (shared tree, and unignored in
`embedders/.gitignore` — `yamnet_context` was also unignored-by-omission, one
`git add -A` from the disappearance that cost `yamnet_aves` its `embedder.py`).
3840-d: `[yam(t-1), yam(t), yam(t+1), aves(t)]`. The YAMNet block is widened and
AVES is not — AVES already integrates the full 1.0 s, and widening it too would
be a second untested lever in one run.

**`02_set/extract.py`** — the fix. Every buffer handed to a context embedder is
padded with `context_frames` frames of **real audio** read from the snip, and
the pad rows are discarded. `SNIP_BUFFER_S` is 30 s, so the pad is available for
every chunk except one butted against a true edge of the source recording —
which is the one place a live deployment also has no neighbour. The rescue path
gets the same treatment and needs it most: a rescue frame is cut alone and
off-grid, so it was previously clamped on *both* sides, and rescue frames are by
construction the ones carrying a labelled event.

Kept frames' audio, roster and labels are byte-identical to the non-context
path (`read_range`/`frame_audio` untouched; the frames are passed into the
padded embed). Only what the embedder can see around them changes.

One benign difference from the `yamnet_aves` cache: `yamnet_aves` feeds YAMNet
one butt-joined buffer of centred crops, so a frame's YAMNet block depends on
which frames sit beside it in the buffer. Padding changes buffer composition, so
centre blocks are not byte-identical to the `yamnet_aves` cache (~1-3%, the
magnitude already documented in that embedder's restore note). It is not
label-correlated, and it makes edge frames *more* deployment-like, not less.

### Worked example, verified by running the real functions

`ins_buzz` at (34, 35): snip (4.0, 65.0); chunk (33.208, 35.800); two frames
written, `[33.208, 34.208]` (0.208 s of buzz — labelled buzz, the threshold is
0.2) and `[34.208, 35.208]`; pads `[32.208, 33.208]` and `[35.208, 36.208]`
embedded then discarded. `expand_chunk` pads 0.8 s each side
(`framelength - overlap_event_s`, ×0.99 leading); the 0.2 is the labelling
threshold, not the padding.

### Verification on `lite` before spending an hour on `medium`

Re-extracted `lite` (10 idents, 37,724 frames, 816 s): **clamped-left 0.0000,
clamped-right 0.0000 for both classes**, width 3840, all finite, 88.7% zeros in
the YAMNet blocks (YAMNet's documented 89.6%), no tiling
(mean `|left - centre|` 0.041).

## Run

```
02_set/main.py --set medium --embedder yamnet_context_aves --workers 0 --verbose   # BUZZDETECT_NO_GPU=1
03_train/main.py --name yavx_h1024 --set medium --embedder yamnet_context_aves \
  --translation general --hidden 1024 --fixed-epochs 150 --verbose -y
```

Comparator is **`yavf_h1024` / `yavf_h1024_r2`** (two-draw mean 0.307) in
`.local/worktrees/yamnet-aves-head-fixed/models/` — the matched fixed-150,
h1024, `yamnet_aves` arm. Not `cv_baseline`.

Branch is based on **main**, with `exp/yamnet-aves-head-fixed`'s `--hidden` and
`--fixed-epochs` patch applied on top — those flags never merged to main and the
matched control was trained with them.

The `--context-frames` / `--context-dims` train-time transform from the first
attempt is left in `03_train/` deliberately: it is how the clamp rate was
measured, and re-running it against the honest cache is the cheapest way to
price the artifact that `context-embedder` still carries.

## Results

## Conclusion
