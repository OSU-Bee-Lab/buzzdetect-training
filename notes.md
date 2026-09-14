# pitchshift-decimate-up-repeat

## Hypothesis

Repeat draw of `pitchshift-decimate-up` (2026-09-14): headline was flat vs
`yamnet_pitchshift` (-0.007 +/- 0.011) but the pre-registered falsifier's named
pattern moved — `1_95` +0.052, `1_114` +0.034, both up, against `1_143`
-0.062 and `1_150` -0.049, both down. Per LOOP.md's hard-fold-gain protocol,
confirm before trusting `1_95`'s move specifically: is this a real fold-shape
effect from removing the tile-seam, or one lucky draw?

## Changes

None. Identical config and cache to `pitchshift-decimate-up`: `--embedder
yamnet_pitchshift_decimate`, `--translation general`, `--fixed-epochs 400`, no
dropout, medium set. Only `--name` differs. No re-extraction — the shared
`medium`/`yamnet_pitchshift_decimate` cache from `pitchshift-decimate-up` was
reused via the worktree symlink.

## Results

vs `yamnet_pitchshift` (x2 rung), both draws side by side:

| fold | r1 delta | r1 delta SD | r2 delta | r2 delta SD | both agree? |
|---|---|---|---|---|---|
| **1_95** | **+0.052** | 0.021 | **+0.024** | 0.014 | **yes, up both** |
| **1_114** | **+0.034** | 0.030 | **+0.043** | 0.036 | **yes, up both** |
| 1_143 | -0.062 | 0.031 | -0.071 | 0.031 | yes, down both |
| 1_150 | -0.049 | 0.048 | -0.023 | 0.046 | yes, down both |
| 1_29 | -0.009 | 0.014 | -0.017 | 0.017 | yes, down both |
| willard/1_11 | -0.020 | 0.029 | -0.016 | 0.022 | yes, down both |
| 53 | +0.009 | 0.021 | -0.002 | 0.023 | no, flips near zero |
| 1_37 | -0.011 | 0.033 | +0.018 | 0.036 | no, flips near zero |

mean sens@fpr0.005 (excl. quiet): r1 0.391, r2 0.393 — headline replicates
tightly (+0.002 between draws). Both draws land at -0.005 to -0.007 vs
`yamnet_pitchshift`'s 0.398, inside the headline delta SD (0.011) both times.

## Conclusion

**Six of eight folds move the same direction on both independent draws**,
including both named hard folds this experiment was designed to check:
`1_95` (+0.052, +0.024 — the era's most stubborn near-chance fold, jet-flyover
failure mode) and `1_114` (+0.034, +0.043). The two folds that disagree
between draws (`53`, `1_37`) are both near zero either way. This is a real,
reproducible fold-shape effect, not a lucky draw: removing
`yamnet_pitchshift`'s tile-seam trades a consistent cost at `1_143`/`1_150`/
`1_29`/willard for a consistent gain at `1_95`/`1_114`, headline flat both
times (-0.005, -0.007, both inside noise).

Read against IDEAS.md's standing hard-fold table: `yamnet_aves` was previously
the only intervention ever to move `1_95` (+0.014/+0.024 over two draws).
`yamnet_pitchshift_decimate` is now a second, independently-confirmed lever on
it, of similar magnitude (+0.024/+0.052) and via an entirely different
mechanism (audio-domain pitch construction, not a different embedder). Given
`1_95`'s failure mode is false positives from a single jet flyover rather than
missed detections, worth noting for whoever next attacks `1_95` (IDEAS items
2/8/15) that this lever exists and stacks with `yamnet_aves`'s mechanism
untested.

Settles item 5b in full: the falsifier does not fire (the hard-fold pattern
does change), but the tile-seam removal is a lateral trade, not a headline
mover — trust `clean` on both entries, on the strength of the two-draw
agreement, not a single caveated run.
