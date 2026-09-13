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

### Verification of the fix itself

**`context_frames == 0` is untouched.** Plain `yamnet` re-extracted over all 10
idents of `lite` under main's `extract.py` and under the patched one: **455
pickles, byte-for-byte identical**, `frametimes.csv` and `annotations.fingerprint`
included (`diff -r` clean).

**The new `yamnet_context` cache is honest**, measured over the same 60 label
pickles in both caches:

| | buggy (`main`) | honest (this branch) |
|---|---|---|
| `left(i) == centre(i-1)` within a pickle | 0.985 | 0.671 |
| edge-clamped rows (`left == centre`) | 105/3501 (3.0%) | **0/3501** |
| frame roster (`frametimes.csv`, 82842 rows) | — | **byte-identical to buggy** |

Same frames, same labels, same timestamps, in the same order. The *only* thing
that changed is what each frame's context block contains. The residual 0.671 is
genuine: a same-label run that really is contiguous audio still has its true
neighbour in the same pickle.

### `context_honest_v1` vs the anchor `cv_baseline_v3`

`--name context_honest_v1 --set medium --embedder yamnet_context --translation
general --fixed-epochs 400 --dropout 0.0` — `context-verify`'s exact config,
8/8 folds, `best_epoch` 400 on every fold in both arms.

| fold | anchor | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| `1_29` | 0.441 | 0.564 | +0.123 | 0.042 | 13 |
| `1_143` | 0.461 | 0.560 | +0.099 | 0.048 | 9 |
| `53` | 0.429 | 0.505 | +0.076 | 0.041 | 11 |
| `1_37` | 0.369 | 0.438 | +0.069 | 0.065 | 5 |
| `1_150` | 0.266 | 0.269 | +0.002 | 0.040 | 14 |
| `willard/1_11` | 0.368 | 0.345 | -0.023 | 0.032 | 12 |
| `1_95` | 0.052 | 0.017 | -0.034 | 0.022 | 23 |
| `1_114` | 0.250 | 0.179 | -0.071 | 0.030 | 13 |

- mean sens@fpr0.005 exclquiet: **0.330 → 0.360 (+0.030 ± 0.015)**
- inclusive: 0.269 → 0.295 (+0.026)
- 5 folds up, 3 down.

### Honest vs buggy — how much of `context-verify` was the leak

| fold | buggy (`context_verify_v1`) | honest | delta | ± SD |
|---|---|---|---|---|
| `1_37` | 0.562 | 0.438 | -0.124 | 0.065 |
| `willard/1_11` | 0.466 | 0.345 | -0.121 | 0.055 |
| `1_143` | 0.640 | 0.560 | -0.080 | 0.033 |
| `1_150` | 0.343 | 0.269 | -0.074 | 0.037 |
| `1_29` | 0.601 | 0.564 | -0.037 | 0.033 |
| `53` | 0.525 | 0.505 | -0.020 | 0.029 |
| `1_95` | 0.017 | 0.017 | +0.000 | 0.014 |
| `1_114` | 0.167 | 0.179 | +0.013 | 0.023 |

- **0.415 → 0.360, -0.055 ± 0.014.** 6 folds down, 1 up, 1 flat.
- So of `context-verify`'s +0.085 over the anchor, **~65% was the leak and ~35%
  survives.** The surviving +0.030 lands almost exactly on the two previous
  honest measurements of this lever: `context-stack`'s rebuild (+0.022) and
  E3 `context-embedder` (+0.022 / +0.061 depending on epoch rule).

### Tiers, total row, same thresholds

| tier | anchor | buggy | honest |
|---|---|---|---|
| `sensitivity_exclquiet` | 0.330 | 0.415 | 0.360 |
| `loud` | 0.778 | 0.870 | 0.793 |
| `untagged` | 0.350 | 0.440 | 0.368 |
| `background` | 0.322 | 0.455 | 0.454 |
| `quiet` | 0.070 | 0.119 | 0.079 |
| `faint` | 0.000 | 0.000 | 0.000 |

**This is the part that changes the reading, not just the size.** `context-verify`
pre-registered its falsifier as *no movement in `untagged` or `loud`*, and
claimed a pass on +0.090 / +0.092. Honestly measured those are **+0.018 and
+0.015** — both smaller than the headline delta and inside their own noise.
What does *not* shrink at all is `background`: +0.132 over the anchor in both
arms. `background` is the 14 long `_background` annotations that live only in
`1_29` and `53`, i.e. continuous drone rather than discrete events — and those
are exactly the two folds carrying the honest headline, along with the two
other rich folds.

So the honest fold signature is **rich folds up, all four hard folds flat or
down** (`1_150` +0.002, `willard` -0.023, `1_95` -0.034, `1_114` -0.071). That
is the same shape `yamnet-aves-context` reported in E2 and recommended against
adopting on, arrived at independently here.

## Conclusion

**The effect does not survive at the size it was logged, and its shape changes.**
`yamnet_context` is worth **+0.030 ± 0.015** over the anchor, not +0.085 — about
one MDE, and consistent with the two earlier honest measurements of the same
lever. Two thirds of `context-verify`'s headline was the label-bucket leak.

It is also no longer readable as a detection gain. Honestly extracted, the
lever pays on continuous background buzz at the two rich folds and does nothing
good at any hard fold, which is where this project's endpoint is. `yamnet_context`
should **not** be the default representation for the era's structural
comparisons; `yamnet_pitchshift` (+0.069 ± 0.011, 8/8 folds up, both hard folds
up) is the era's real lead and is unaffected by this bug.

Standing facts corrected: `IDEAS.md`'s "1a done, +0.085" and the anchoring of
`hidden-context-verify` to 0.415. `log.jsonl` entries amended rather than
rewritten (see `main`).

## Still open

- **`yamnet_context_aves` needs the same treatment.** Its shipped `medium`
  cache was built on `exp/yamnet-aves-context` (2026-09-10) with the padded
  extraction, so it is honest — but it predates the **2026-09-11 annotation
  revision**, so its fingerprints no longer match. The next run that touches it
  on `main` would re-extract the stale idents *the buggy way* unless this fix is
  merged first. Measured on that cache now: clamp rate 0/3501 and in-pickle
  adjacency 0.634, i.e. currently honest. Do not train on it from `main` before
  merging this branch.
- **A clean re-extraction of any embedder currently fails `read_fold_roles`.**
  `folds.csv` lists `.../2026-07-27/1_99` as a `train` fold, but the current
  `annotations.csv` has no rows for that ident, so `extract_set` never visits it
  and never creates its directory; `03_train/dataset.py::read_fold_roles` then
  raises `FileNotFoundError`. Every embedder cache on disk only passes that
  check because it carries an empty leftover `1_99/` directory from a
  pre-revision build. This run reproduces the original by creating the same
  empty directory. Pre-existing on `main`, unrelated to this fix, and it will
  bite the next embedder extraction.
- One draw only. There is no seed control; a repeat would cost one CV.
