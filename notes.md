# perch-centred

## Hypothesis

IDEAS.md item 2 (queue, ranked #1 of the actionable items -- the only queued
idea not blocked by the AVES injunction). `perch-probe` (E3) found Perch v2's
apparent gain on its native 5.0 s frame tracked frame-length dilution of the
labels, not the embedder (`overlap_event_s` scales with framelength, so a
5 s frame needed 1.0 s of label overlap against YAMNet's 0.192 s) --
inconclusive, not a verdict. A later attempt to reuse that 5 s cache by
broadcasting rows onto the lead's 0.96 s frame grid (`perch-broadcast-join`)
was closed forbidden: it duplicates one Perch row across many frames near it
and zero-fills frames with no nearby real row, the same inflation mechanism
as `context-stack`'s cached-row leak.

This is the compliant design item 2 calls for: Perch on a real 5.0 s window,
**centred** on each 0.96 s frame of the lead's own grid, extracted for real
(no cache join). Every row is Perch's own forward pass over real, contiguous
audio nobody else has embedded -- frame count, frame grid and labels are all
byte-identical to `yamnet`/`yamnet_trunk`/etc., so the headline joins
`cv-baseline-v3`-family numbers directly (unlike `perch-probe`).

Motivation for Perch specifically: `diagnostics/2026-09-09_jet_1_95` measured
Perch's linear separation of the `1_95` jet-vs-buzz confusion far ahead of
YAMNet or AVES (P(jet outranks buzz): yamnet 0.860, aves 0.558, **perch
0.239**) on the era-open lead. `1_95` is still the lowest-sensitivity fold on
every representation tried since, and its threshold is still set by aircraft
frames (IDEAS.md's hard-fold census). If Perch's bioacoustic training domain
carries real separating signal for that confusion, a real extraction on the
matched frame grid should show it -- via a linear head, frozen, no
fine-tuning (same regime `perch-probe` used).

**Falsifier (pre-registered, per item 2): `1_95` is the test, not the
headline.** A headline gain whose `1_95` delta sits inside its own eval-
sampling SD is a rich-fold gain, not evidence Perch helps the jet confusion.
Given the context/averaging prior (four for four: context levers gain on rich
folds and leave `1_95`/`1_114`/`1_150` flat-to-down) -- and this design DOES
read wider audio around each frame, so it must be checked against that prior
even though the mechanism (one real forward pass on a real wide window,
frozen probe, not a stacked/averaged embedding) differs from every context
lever tried so far.

## Changes

New embedder `embedders/perch_centred/embedder.py`: `framelength_s=0.96` (the
lead's grid, not Perch's native 5.0 s), `samplerate=32000` (Perch's native,
avoids a second resample), `context_frames=3` (2.88 s padding each side,
covering the 2.02 s needed for a full centred 5.0 s window even at a chunk's
first/last frame -- `02_set/extract.py`'s existing contiguous-audio mechanism,
the same one `yamnet_context`/`yamnet_trunk_context` use, not a cache join).
`embed()` slices a real 160000-sample (5.0 s @ 32 kHz) window centred on each
frame directly from the padded contiguous buffer it's handed, applies Perch's
own `normalize_audio` (DC-remove, peak-scale to 0.25), and batches through the
same SavedModel `embedders/perch` already loads.

Two-interpreter split, exactly as `perch-probe`: stage 2 (extraction) runs
under `.local/venv-perch-extract` (TF 2.21, sees no GPU -- CLAUDE.md,
CPU-only), stage 3 (training) under the pinned conda env (TF 2.16.2), since
stage 3 never runs the embedder (`initialize=False`, reads `n_embeddings` and
pickles off disk).

```bash
BUZZDETECT_CHUNK_FRAMES=32 .local/venv-perch-extract/bin/python -u 02_set/main.py \
  --set medium --embedder perch_centred --workers 2 --verbose
/home/luke/anaconda3/envs/buzzdetect-train/bin/python -u 03_train/main.py \
  --name perch-centred --set medium --embedder perch_centred --translation general -y
```

Comparator: `cv-baseline-v3`-family anchor (frame grid, labels and budget all
match, unlike `perch-probe`) -- a fresh refresh if the era's annotations have
moved materially since the last refresh (`cv-baseline-v3-refresh`,
2026-09-16).

## Results

**Blocked before a single medium ident completed.** No CV was run; there is
nothing to compare. This section records the engineering investigation so
the next agent doesn't redo it.

**The embedder itself works correctly.** Verified directly (outside the
pipeline): `embed()` returns the right shape (n, 1536) for arbitrary input
lengths, and a fresh lite extraction produces real Perch embeddings for the
first ident's first snip.

**Memory: found and partly fixed one real bug, a second remains
undiagnosed.**

1. `self.model` (a `tf.saved_model` signature) retraces/recompiles under XLA
   JIT (its default) whenever it's called with a batch size it hasn't seen
   before, and TF never releases an old trace's compiled graph. `extract.py`
   hands this embedder chunks of varying size (context-padding shrinks near
   an ident's edges, rescue calls pass n=1, a snip's last chunk is ragged),
   so the raw call pattern produces many distinct shapes over a real
   extraction -- unbounded host RSS growth (measured: 2 GB -> 16+ GB within
   under a minute on `lite`'s first ident, independent of `CHUNK_FRAMES`).
   **Fix 1** (`embed()`): pad every call up to a multiple of `_BATCH_BUCKET`
   (16), capping the distinct shapes at a handful. This slowed the growth
   but did not stop it.
2. Isolating just the repeated-call pattern (a standalone script, same
   `(16, 160000)` batch every time, no extraction pipeline) showed RSS is
   NOT flat under default JIT either way past a certain point, and IS flat
   (2.7 -> 2.9 GB over 20+ calls, ~7 MB/call trailing off) once XLA JIT is
   disabled (`tf.config.optimizer.set_jit(False)`, set before the model
   loads) -- at the SAME per-window throughput `perch-probe` already
   measured (~0.4 s / 5 s window), so JIT was not buying real speed on this
   workload. **Fix 2** (`initialize()`): disable JIT.
3. **With both fixes applied, RSS still grows inside the real pipeline** --
   slower than before (roughly 3 GB/min vs the original much-faster blowup),
   but real and sustained across chunks of the SAME snip (not just across
   shape changes), which Fix 2's isolated repro did not reproduce. The
   isolated test calls the model directly in a tight loop; the real
   extraction also runs `read_range`/`_read_pad_frame` (soundfile reads +
   librosa resampling) many times per chunk for the context padding. That
   path is untested in isolation and is the leading remaining suspect --
   not confirmed.

**Compute time, separately from the memory question:** at Perch's own
established CPU rate (~0.4 s / 5 s window, `perch-probe`), and this design's
~5.2x higher forward-pass density (0.96 s frame grid vs Perch's native 5.0 s,
so ~5.2 full-window passes per second of audio instead of ~1), a full
`medium` extraction is roughly 5x `perch-probe`'s own ~1.5-3 h estimate --
**~8-15 h**, before accounting for any restart overhead a memory-safety
workaround would add. Long, but not by itself disqualifying, if the memory
question were resolved.

**What was tried and ruled out as the primary fix:**
- A burst-and-restart driver (`run_perch_centred.sh`, committed): kills and
  relaunches the extraction process every `BURST_S` seconds, relying on
  02_set's ident-level fingerprinting to resume safely. This *works* as a
  safety net (verified: a killed burst resumes cleanly, memory resets to
  ~2.5 GB on restart) but is not viable as the actual strategy here --
  `extract_ident_both` only writes an ident's output at the very end (after
  all its snips), so a kill mid-ident discards that ident's progress
  entirely. A single snip has taken several minutes of wall time without
  finishing even at `CHUNK_FRAMES=4-16`, so a burst short enough to stay
  memory-safe (under ~5 min, by the growth rate above) cannot reliably
  complete even one snip, let alone the up-to-22-snip idents this set has --
  net forward progress would be near zero, not just slow.

## Conclusion

**Not run; blocked on an unresolved (partially diagnosed) host-memory
growth in repeated CPU calls to Perch's SavedModel signature**, on top of
this design's already-long compute-time budget (~8-15 h). Two real fixes
landed (batch-shape bucketing, disabling XLA JIT) and are kept in
`embedders/perch_centred/embedder.py` since both are correct improvements on
their own terms, but neither fully stops the growth inside the real
extraction pipeline. The leading untested hypothesis is the repeated
`read_range`/`_read_pad_frame` (soundfile + librosa resample) calls the
context-padding mechanism makes per chunk, not the model call itself.

**Recommended next step, concretely:** isolate `read_range`/`_read_pad_frame`
the same way the model call was isolated above (call them in a tight loop
against a fixed audio file, watch RSS) to confirm or rule them out before
looking anywhere else. If they're clean too, the next thing to try is
running each chunk's model call in a short-lived subprocess (`multiprocessing`,
one-shot, result over a pipe) so OS process teardown guarantees the memory is
reclaimed between calls regardless of what's holding it -- trading repeated
model-load overhead (SavedModel load is not free) for guaranteed safety, and
only worth it if the per-call overhead stays small next to Perch's own
~0.4 s/window cost.

Handed off via `HANDOFF.md` rather than pushed through, given the time
already spent diagnosing two real bugs without reaching a stable extraction --
this needs fresh budget, not persistence on the same three hypotheses.

## Fix (batch-11 fixer, 2026-09-22)

**Cause: the whole annotation chunk went through Perch as one batch.** The
context path (`extract.py::_embed_with_context`) hands `embed()` every frame
of an annotation chunk at once, up to a full 360 s snip (~375 windows plus
pads). `BUZZDETECT_CHUNK_FRAMES` only chunks the non-context path, which is
why no `CHUNK` value helped. `smoke_jitfix.log` shows it directly: a single
`Allocation of 6146631936 exceeds 10% of free system memory` on the first
snip, then exit 137. `_BATCH_BUCKET` rounding kept that one giant call and
turned every distinct chunk length into another shape. The "slow growth"
in result 3 was this, not `read_range`.

**Fix** (main, `embedders/perch_centred/embedder.py`): `embed()` now runs the
model in fixed `(16, 160000)` batches, the last zero-padded with its pad rows
dropped. Rows are unchanged: each window is still normalized on its own.
JIT stays off.

**Verified** on `lite` (`smoke_batchfix.log`, `--workers 1`, CPU): worker RSS
held at 2.5-2.8 GB over 6 snips / 14 min, falling back at snip boundaries.
Before the fix it passed 16 GB within a minute. Throughput is ~1.2-2 min per
360 s snip. `run_perch_centred.sh` (the burst driver) is no longer needed.
