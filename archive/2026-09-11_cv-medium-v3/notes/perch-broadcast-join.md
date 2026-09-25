# perch-broadcast-join

## Hypothesis

IDEAS item 2, sub-item 2a. Perch 2.0 is the strongest linear-probe encoder
measured on the jet-vs-buzz separation diagnostic (perch 0.239 vs yamnet
0.860, aves 0.558 — lower is better), and `1_95`'s threshold is still
aircraft-set on the era lead. `perch-probe`'s earlier E3 negative died of a
frame-length confound (Perch's 5 s frame forced `overlap_event_s` up, diluting
labels), so this broadcasts Perch's already-extracted 5 s embedding onto the
lead's 1.0 s frame grid instead of re-deriving Perch's own frame grid: no
re-extraction, no label dilution, same frame count and labels as the lead.

Falsifier (from IDEAS.md): `1_95` is the test, not the headline. A headline
gain with `1_95` inside its own delta SD is a rich-fold gain, indistinguishable
from the broadcast just adding ~2.5 s of context each side (the honest
`yamnet_context` pattern: rich folds up, hard folds flat).

## Changes

New embedder `yamnet_pitchshift_aves_mid_perch` (5888-d =
`yamnet_pitchshift_aves_mid`'s 4352-d + Perch's 1536-d), built by
`tools/join_broadcast.py` rather than extract.py: both source caches already
exist for `medium`, so the script reads each ident's `frametimes.csv`
(the durable row->timestamp map) from both trees and matches every
`yamnet_pitchshift_aves_mid` frame to the Perch frame whose 5 s window covers
its centre, instead of assuming a clean `j -> j // 5` grid.

That assumption doesn't hold: chunk boundaries depend on `framelength_s` and
`overlap_event_s` (which scales with it — `expand_chunk`), so the two
embedders' chunks aren't nested. Worse, Perch's 5 s frame makes
`extract_ident_both` skip any annotation-cluster snip under 5 s outright ("snip
too short for embedder"), so isolated short buzz events can have **no** Perch
frame anywhere nearby. A naive nearest-neighbour join found matches up to 3601 s
away for 2.65% of frames before this was caught (measured across all 79,952
medium frames). Fixed by zero-filling the Perch block whenever the nearest
candidate's centre is farther than `coarse_len/2 + fine_len/2` (3.0 s) from the
fine frame's centre — 2297 frames (2.87%) zero-filled; the remaining matches
average a 1.28 s centre-gap, max 2.99 s (i.e., genuine containment/near-overlap
only).

Training: same architecture and budget as the current era lead
(`epoch-budget-700`, `--fixed-epochs 700`), only the embedder changes.

```bash
tools/join_broadcast.py --set medium --fine yamnet_pitchshift_aves_mid --coarse perch --out yamnet_pitchshift_aves_mid_perch
03_train/main.py --name perch-broadcast-join --set medium --embedder yamnet_pitchshift_aves_mid_perch --translation general --fixed-epochs 700 -y
```

## Results

<tools/results.py output pending>

## Conclusion

<pending>
