# cv-baseline-v3-refresh

## Hypothesis
`cv-baseline-v3` (the era anchor, trained 2026-09-11) turned out to be stale
relative to current annotations: ongoing human annotation work in `01_annotate/
Even Sample` (the era-cutover revision to `combine.R`, commit f0ac4eb, landed
2026-09-11 15:41, then the corpus kept growing for ~2 more days) roughly
doubled-to-tripled `buzz_events_exclquiet` on every fold by 2026-09-13/14.
Discovered while comparing a freshly-extracted embedder (`asym-context-yamnet`,
item 10) against `cv-baseline-v3` directly: the paired "buzz events" counts in
`tools/results.py`'s report were ~40% of what a same-day comparison against a
recent model (e.g. `aves-mid-repeat`) showed for the same folds.

This reruns the anchor's *exact* config (bare YAMNet linear probe, no
dropout, `--fixed-epochs 400`, translation general, medium set, 8 rotating
folds) against the current, already-shared `yamnet` embedding cache -- no
extraction needed, since the cache is additive and other experiments have
kept it current. Purpose: a fresh, comparably-dated anchor for any future
plain-YAMNet-family comparison, and a diagnostic on whether the anchor number
itself moved.

## Changes
None -- identical config to `cv-baseline-v3`, just rerun today.

## Results
Per-fold `buzz_events_exclquiet` grew 1.5-3x on every fold (13->32, 11->28,
12->26, 9->22, 14->21, 23->46, 5->14, 13->28 for
1_29/53/1_11/1_143/1_150/1_95/1_37/1_114) yet the headline barely moved:
`sensitivity_exclquiet` 0.330 -> 0.329 (-0.001), inclusive 0.269 -> 0.268.
Full per-fold breakdown is in `models/cv-baseline-v3-refresh/folds_sx.csv`.

## Conclusion
The anchor's headline number is robust to the annotation growth -- 0.330 was
not an artifact of a small, unrepresentative buzz-event pool, which is
reassuring for every experiment that already compared cleanly against the
original `cv-baseline-v3` on the SAME (stale) vintage (their deltas are
internally valid; see log entries for `context-verify`, `yamnet-aves-verify`,
`hidden-head-verify`, `dropout-verify`, `binary-control` -- all confirmed
matched-vintage by diffing `buzz_frames`/`buzz_events_exclquiet` against their
own stored `cv_baseline_v3` copy).

**What is NOT fine**: any comparison that paired a model trained on the
*current* (grown) annotation set against the *original* `cv-baseline-v3`
mixes vintages and inflates or distorts the delta and its per-fold SDs, even
though the headline itself barely shifted (per-fold event counts, and
therefore per-fold noise and any tier-level reading, did move). Confirmed
affected by diffing each entry's own model's `buzz_frames` against its stored
`cv_baseline_v3` copy: `yamnet-pitchshift`, `context-frames-fix`,
`yamnet-pitchshift-down` (all 2026-09-13, all trained on the grown set,
compared against the original 2026-09-11 anchor). Those three entries are
being amended separately with corrected deltas against this refreshed anchor.

This model (`cv-baseline-v3-refresh`) is the correct comparator for any future
plain-YAMNet-family experiment; prefer it over `cv-baseline-v3` going forward.
No further re-anchoring needed unless a similarly large annotation shift
happens again -- watch for it via the same buzz-event-count diff, not by
assumption.
