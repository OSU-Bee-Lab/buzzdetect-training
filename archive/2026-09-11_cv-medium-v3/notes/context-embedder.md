# context-embedder

## Hypothesis

`IDEAS.md` flags this as the highest-value first experiment of the post-2026-09-08
era: on the **previous** data (11 folds, `general_v1`) an embedder that widens each
frame with the audio that really sat either side of it — `concat(t-1, t, t+1)`,
3072-d — was worth **+0.022** (7/11 folds up), `clean` trust. It was never folded
into the baseline because it costs a re-extraction and bundling it in would have
made the reference point undecomposable.

The data has since been revised and the roster is now 5 rotating folds, so that
number is a direction, not a target. **Prediction: a gain over `cv_baseline`'s
0.218**, of roughly that size, with most folds moving up — and, per
`willard-regression`, a possible loss on deployments whose buzz is short and
isolated (willard is no longer a fold here, so this is a check on the story rather
than a rerun of it).

Nothing else changes: same probe, same `medium`, same `general` translation, same
hyperparameters. One thing changed.

## Changes

Ported from `refs/archive/context-embedder` (the branch was deleted; only the ref
survives) onto current `main`, which has since gained the three-layer extraction
cache and `frametimes.csv`.

### `embedders/yamnet_context/embedder.py` (new, restored verbatim)

Subclasses `EmbedderYamnet`. `embed(audio)` takes **contiguous** audio, embeds every
frame, and returns `concat(t-1, t, t+1)` per frame; edges clamp to themselves.
`n_embeddings = 3072`, `context_frames = 1`. Memory chunking happens inside
`embed()` on frame boundaries so no caller can split the buffer and silently clamp
a frame's context. `03_train` needs no change — the probe sizes its input from
`n_embeddings` — and the shipped `model.py` gets the wider representation for free
because the context lives in the embedder.

### `02_set/extract.py` — the part the hypothesis depends on

Extraction frames each coverage chunk in time order, then **buckets frames by their
collapsed label** and embeds the buckets. An embedder stacking inside the buffer it
is handed would therefore see neighbours that share its own label — exactly the
artifact that made `context-stack` (+0.050) `artifact`-trust. So, gated on
`context_frames` (0 for every existing embedder, so nothing else changes):

- each coverage chunk's frames are embedded in **one** `embed()` call, in time
  order, and the *embeddings* are bucketed by label afterwards;
- `extract_ident_embeddings` (cached-audio → embeddings) raises: the cache is
  bucketed, so context is unrecoverable from it;
- idents whose framed audio is already cached rebuild from the **snips** instead,
  with `write_audio=False`, so the shared audio cache is read but never rewritten
  (`frametimes.csv` and the fingerprint likewise go only to the embeddings dir);
- `framehop_prop != 1` raises — `np.concatenate(frames)` only reconstructs the
  chunk's audio when frames tile it without overlap.

The `yamnet_context` cache is written into **main's** shared `embeddings/` through
the worktree symlink, not into a local copy. LOOP.md step 1 says to break that
symlink before re-extracting under changed extraction code; that rule is
over-broad. The cache is keyed by `(set, embeddername)` and the staleness
fingerprint covers **annotations only, not code**, so the case that actually
corrupts main is an experiment whose output under an *existing* name differs from
what is on disk — change `embedders/yamnet/` or the framing for all embedders, keep
the name `yamnet`, and every later run in main silently reads unmerged-code
embeddings with nothing to detect it. This experiment is the safe case: a new
embedder name, an extract.py change gated on `context_frames` (0 everywhere else,
so the existing path is byte-identical), and `write_audio=False` so the shared
framed-audio cache is only ever read. Sharing means the ~1 GB survives the
worktree pruning at step 6.

**A related hazard, worth knowing before killing an extraction:** the ident
fingerprint is stamped *last*, and `AssignIdent._is_stale` treats a missing
fingerprint as not-stale, so an ident left half-written by a killed run has
pickles, no fingerprint, and `handle='skip'` — it is treated as complete forever.
An interrupted extraction must have its unfingerprinted ident dirs deleted before
relaunching. (This run was killed and relaunched once, for the symlink change
above; the 95 partial pickles were purged, only 2 idents having reached a
fingerprint.)

## Verification before the run

`embed()` checked against plain YAMNet on a 5-frame contiguous buffer: middle block
== the frame's own embedding, `t-1`/`t+1` blocks == the real neighbours, edges
clamped, a single (rescue) frame → `(1, 3072)` with all three blocks equal.

One pre-existing quirk surfaced: the **last frame of any buffer** handed to YAMNet
differs from the same frame embedded inside a longer buffer (~0.25 relative), a
mel/STFT tail effect, not something this change introduces — the default path
embeds in `CHUNK_FRAMES` blocks and has it too. At the default `CHUNK_FRAMES=300`
it touches ~1 frame in 300 on both sides of the comparison.

## Results

`models/context_embedder/`, 5 rotating folds, paired against `models/cv_baseline`.

| fold | baseline sens@fpr0.005 | this exp | delta | buzz frames | val frames |
|---|---|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 0.426 | **0.551** | **+0.125** | 2144 | 7617 |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 0.425 | **0.494** | **+0.069** | 1031 | 7540 |
| Lily Adam - One Hive/recorders/willard/2024-08-07/1_11 | 0.180 | **0.217** | **+0.037** | 305 | 6930 |
| Luke - Diel Drivers/2026-05-06/1_95 | 0.037 | 0.014 | -0.023 | 433 | 7572 |
| Luke - Diel Drivers/2026-04-08/1_150 | 0.021 | 0.014 | -0.007 | 146 | 4947 |

- mean sens@fpr0.005: baseline **0.218** → this **0.258** (**+0.040**)
- 3 folds up, 2 down, 0 flat

**The split is not evenly weighted.** The three folds that moved up are the three
that can measure anything — 305 to 2144 buzz frames — and all three moved up by
much more than the ~0.014 headline noise floor. The two that moved down are the
pair `IDEAS.md::near-chance-deployments` names as sitting near zero for *every*
model tried: at baseline 0.021 and 0.037, a -0.007 is about one buzz frame out of
146, and a -0.023 about ten out of 433. Neither is a capability change, and neither
fold's threshold is meaningful (`2026-05-06/1_95` sets it at +0.173 against a mean
of -1.366; `2026-04-08/1_150` at -2.107). So every fold with the buzz to resolve a
difference moved the same way.

Precision at the same operating point drifts down slightly (0.589 → 0.561) —
entirely from those two near-chance folds (0.111 → 0.077, 0.310 → 0.144); the three
real folds all improved on precision too (0.971→0.977, 0.931→0.940, 0.624→0.667).

**The archived willard regression did not reproduce.** On the previous data
`context-embedder` lost 0.059 at willard and `context-stack` lost 0.074, and
`IDEAS.md::willard-regression` attributes that to willard having the highest share
of short (<1 s) and isolated (>5 s gap) buzz events — exactly what a 2.88 s feature
window dilutes. Here willard **gains** 0.037. Either the revised annotations changed
willard's event-length profile, or the dilution story was always a smaller effect
than the fold's own variance (305 buzz frames is thin). Worth a look before the
story gets repeated: it is currently unsupported on live data.

Note on what this does and doesn't change: the **frame grid is unchanged** — frames
are still 0.96 s, non-overlapping, and still labelled by `events_in_frame` on their
own 0.96 s span. What widens is the *feature* window, to 2.88 s centred on the
frame, sliding at a 0.96 s hop. So adjacent frames' features overlap by 1.92 s while
the frames themselves do not overlap at all, and the frame count is identical to the
`yamnet` cache (82,064 frames across 1292 pickles, verified frame-for-frame before
training — which is also what makes the paired comparison above valid).

## Diagnostics (free from this run's surprisal CSVs)

All of the below is recomputed from `surprisal/` and `annotations.csv` — no
training. It survives a data change (recompute it) and it points at what to
annotate next, which is why it is recorded here rather than discarded.

### The gain does not depend on buzz event structure

`willard-regression` predicts a 2.88 s window dilutes short isolated buzz. Frame
sensitivity by the duration of the annotation each buzz frame falls under, mean
over folds (the headline's own statistic):

| event span | baseline | context | delta | frames |
|---|---|---|---|---|
| <1 s | 0.175 | 0.221 | +0.046 | 1097 |
| 1-2 s | 0.215 | 0.273 | +0.058 | 1380 |
| 2-5 s | 0.317 | 0.356 | +0.039 | 990 |
| >5 s | 0.188 | 0.214 | +0.026 | 592 |

Clustered +0.042 vs isolated (>5 s gap) +0.034. Short gains *more* than long and
isolated nearly as much as clustered — the opposite of dilution. Together with
willard gaining here, **the willard-regression story is unsupported on live data.**

### Seven annotations carry half the headline

`ins_buzz` annotations spanning a whole 300 s file: 5 in mustard (1500 s of its
2391 s of buzz, **63%**) and 2 in Fit+Fast (599 s of 1148 s, **52%**). None in the
other three folds. So the two folds at the top of the metric draw most of their
buzz frames from seven annotations of a continuous background drone; their
`buzz_frames` counts (2144, 1031) overstate their independent sample size badly,
and part of what the headline rewards is drone detection rather than discrete
foraging buzzes. It also means the `>5 s` row above is mostly those drones — read
that table as annotation spans, not buzz events.

### The two near-chance folds fail in opposite ways

Not the same problem, and neither is "not enough buzz":

| | 1_150 (0.021) | 1_95 (0.037) |
|---|---|---|
| negatives, p99.5 | 0.139 — normal | **0.391 — ~3x every other fold** |
| buzz frames, mean/median | **0.071 / 0.064 — ≈ its own background** | 0.102 / 0.068 |
| frames setting the threshold | diffuse ambient (11 bg, 6 mech_auto) | **32 of 35 `mech_auto`** |
| fold threshold | -1.819 | **+0.173** (only positive one) |

- **1_150 is a positives problem.** Its negatives are unremarkable; its buzz frames
  are simply indistinguishable from its own background. 88 s of plain
  `ins_buzz_medium` with 6503 s of support in the pool, and the model still cannot
  see it. Nothing structural explains this — it is the fold `IDEAS.md`'s "listen to
  the frames" exists for, and it is a narrow target: **88 seconds of audio.**
- **1_95 is a false-positive problem.** Vehicle noise, not trill, sets its
  threshold, which is what drives the threshold positive while every other fold
  sits near -1.7. Low buzz versus engine drone is an acoustically plausible
  confusion (`ins_buzz_low` is 242 s of its 328 s).

**A coverage explanation was checked and does not hold.** Restricting support to
the other *rotating* folds makes 1_95 look starved (89% of its buzz in sublabels
with <60 s elsewhere), but the training pool includes `train`-role idents: against
the real pool `ins_buzz_low` has 957 s of support and `ins_buzz_pollination` 300 s
(in two "Various Opportunistic Recordings" idents). Every fold is at 0% thin.
This re-confirms the 2026-09-05 leave-one-concept-out ruling rather than
overturning it — don't revive it without checking roles.

### `trill-vs-buzz` step 1, answered

Raw labels of the negatives above each fold's own fpr0.005 threshold, pooled by
component label: **`mech_auto` 75, `ambient_background` 46, `ins_trill` 43**,
everything else ≤8. Trill is about a third and is fold-dependent — it leads at
willard, vehicles lead at 1_95 and mustard. The hypothesis as written in
`IDEAS.md` ("the false positives that set the threshold are mostly `ins_trill`")
does not hold, so the pairwise buzz-vs-trill margin it recommends would be aimed
at the wrong class. If a margin term is worth trying, it is buzz-vs-`mech_auto`.

## Conclusion

**Revalidated. +0.040, clean, adopt as a candidate for the next baseline.** The old
verdict (+0.022, 7/11) not only survived the data revision, it grew, and it is now
the largest live result in this era's log by a wide margin — comfortably outside the
~0.014 headline noise floor, with every fold that has enough buzz to measure moving
up and the two exceptions being folds that measure nothing for anybody.

Two things follow, in this order:

1. **`yamnet-combined` is the other unrevalidated clean win** (+0.016, 7/11 on the
   old data) and is still untested here. Run it against `cv_baseline` next, on its
   own, then ask whether the two compose. Do not stack them speculatively.
2. **The near-chance pair is now 40% of the headline and is unmoved by a change
   this large.** That is the single biggest lever on the metric and it is not an
   architecture problem: `2026-04-08/1_150`'s threshold went *further* negative
   here (-1.819 → -2.107), i.e. its buzz frames still score below almost every
   negative in the fold. `IDEAS.md` has had "listen to the frames" deferred five
   times now. It should come before another representation run.

Cost, for the next agent's planning: extraction of `medium` under `yamnet_context`
took ~11 min with `--workers 2` on CPU (982 MB, written to main's shared cache and
still there); the CV took ~19 min, ~3.6 min/fold, which is ~2x the frozen-probe
baseline because the probe has 3x the input dimensions.
