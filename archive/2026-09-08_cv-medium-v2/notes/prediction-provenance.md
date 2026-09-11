# prediction-provenance

## Hypothesis

`folds/<fold>/predictions.csv` carries only `activation_ins_buzz` and
`correct` (ground truth, not correctness) — no ident, no frame index, no raw
label. That blocks "what are the ~22 negative frames above threshold?",
"which kind of buzz do we miss?", and any event-level read, which is exactly
what sank `framehop-overlap`'s interpretability. Adding provenance columns
should be a pure instrumentation win: no metric change, no retraining needed
to benefit later comparisons.

## Changes

- `03_train/train.py::_score_fold` now also emits `path` (the source
  pickle — one event, of no fixed depth, so this is the join key rather than
  a parsed-out ident), `frame_index` (0-based position within that event's
  frame sequence), and `labels_raw` (the raw `+`-joined label combination,
  unaffected by translation unlike `correct`). Existing columns and their
  meaning are untouched — additive only, so `03_train/metrics.py`,
  `03_train/sx.py::_fold_sens`/`summarize_folds`, and `resummarize.py` all
  read predictions.csv exactly as before.
- `03_train/sx.py::read_fold_predictions_events` — collapses the frame-level
  table to one row per (fold, event): max activation (did any frame
  overlapping this event fire), first `correct`/`labels_raw`, frame count.
  Returns `None` for models trained before this landed (no `path` column) —
  checked directly, not inferred from an exception.

**Scope cut from the original idea:** IDEAS.md described joining
`frame_index` to `frametimes.csv` for an absolute timestamp. `frametimes.csv`
is written by `02_set/extract.py` but isn't actually on disk for any ident in
`medium` or `lite` yet — it was added after these were last extracted, and
the annotation fingerprint hasn't changed since, so nothing re-triggered it.
Backfilling it means a re-extraction, which this loop iteration deliberately
skipped (short, no re-extraction, no retraining). `path`+`frame_index`
already answers "which frames" and "which event" without it; timestamps are
a follow-up once a fold re-extracts for an unrelated reason.

Also scope-cut: **old models' `predictions.csv` are not backfilled.** The
new columns are written only going forward, by a training run that calls the
model. `resummarize.py` only reads existing `predictions.csv` to rebuild
`folds_sx.csv` — it never re-scores, so it cannot add columns a training run
didn't write. A separate rescore-from-`model.keras` script would be needed to
backfill old models; not built here since nothing pooled requires it yet.

## Results

No CV run — this is a code change, verified without retraining or
re-extracting:

- `_score_fold` run directly against a real `medium`/`yamnet` fold
  (`JamesU - MustardBumbler/1_29`, `general_v1` translation) with a
  freshly-initialized (untrained) model: 6984 frames, 45 distinct events,
  `frame_index` resets to 0 at every event boundary and is contiguous within
  one, `path`/`labels_raw` constant within an event.
- `read_fold_predictions_events` against that same scored fold: correctly
  collapses to 45 rows, `n_frames` matches each event's frame count.
- `read_fold_predictions_events` against a synthetic old-format
  `predictions.csv` (no `path` column): returns `None` rather than raising or
  silently mis-grouping. `read_fold_predictions` (frame-level) is unaffected
  either way.

## Conclusion

Landed as designed, scoped down to what's actually free right now: ident
(via `path`) and raw labels are on every future `predictions.csv` at zero
extra cost, and `trill-vs-buzz`'s diagnostic step 1 ("tabulate the raw labels
of negatives above threshold") is now a query on the next model trained,
frozen or fine-tuned — no separate run. The frametimes/timestamp join and
old-model backfill are real gaps but weren't free the way the ident+labels
part is; leaving them for whenever a re-extraction or a rescore script
happens for an unrelated reason, rather than spending a loop iteration on
either now.

No `sens@fpr0.005` number to report — this doesn't touch training or
scoring math, only what's recorded alongside it.
