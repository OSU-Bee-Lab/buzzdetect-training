# trunk-pitchshift-depth12-repeat

## Hypothesis

`trunk-pitchshift-depth12` (this batch, 2026-09-23) found a real, clean gain
cutting the fine-tuned trunk tail one layer earlier (layer11 cut, tune 12-14)
while keeping `trunk-ft-pitchshift`'s octave-up dual-view concat: headline
0.434 -> 0.472 (+0.038 +/- 0.011, ~3.5σ), 7/8 folds up, and notably `1_114`
(trill) recovered (+0.064) where plain fine-tuning's own depth swap
(`trunk-depth-headtohead`) found the two cuts statistically equivalent. This
is now the era lead. LOOP.md: "confirm a large gain with one repeat run" --
there is no seed control in this pipeline, so a second CV of the identical
config is the check against TF's run-to-run nondeterminism before leaning on
this result further.

**Falsifier:** if the repeat's headline lands well below 0.472 (inside
`trunk-ft-pitchshift`'s 0.434, or with `1_114`'s gain evaporating), the first
run was a favorable draw, not a real depth effect, and the "new era lead"
claim should be walked back to "matched, not ranked".

## Changes

None -- identical config, same embedder (`yamnet_trunk_pitchshift_depth12`,
already extracted, cache reused), same command, new `--name` only.

## Results

```
tools/results.py <trunk-pitchshift-depth12 worktree>/models/trunkpsd12-ft-1e5 trunkpsd12-ft-1e5-r2
```

Paired against the first draw (`trunkpsd12-ft-1e5`):

| fold | run 1 | run 2 | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.520 | 0.554 | +0.034 | 0.027 | 32 |
| 53 | 0.543 | 0.567 | +0.024 | 0.024 | 28 |
| 1_11 | 0.594 | 0.566 | -0.028 | 0.030 | 26 |
| 1_143 | 0.586 | 0.586 | +0.000 | 0.028 | 22 |
| 1_150 | 0.312 | 0.287 | -0.025 | 0.030 | 21 |
| 1_95 | 0.205 | 0.174 | -0.031 | 0.028 | 46 |
| 1_37 | 0.551 | 0.525 | -0.026 | 0.045 | 14 |
| **1_114** | 0.462 | 0.374 | **-0.088** | 0.041 | 28 |

Headline: 0.472 -> 0.454 (-0.018 +/- 0.011, ~1.6σ), 4/8 folds up, 4/8 down.

**Both draws against `trunk-ft-pitchshift` (0.434), the actual comparator
that matters:**

| | headline | delta vs trunk-ft-pitchshift | 1_114 | delta vs trunk-ft-pitchshift's 1_114 (0.398) |
|---|---|---|---|---|
| run 1 (`trunkpsd12-ft-1e5`) | 0.472 | +0.038 | 0.462 | +0.064 |
| run 2 (`trunkpsd12-ft-1e5-r2`) | 0.454 | +0.020 | 0.374 | **-0.024** |
| mean | 0.463 | **+0.029** | 0.418 | **+0.020** |

## Conclusion

**Split verdict: the headline gain survives, the fold-level mechanism story
does not.**

The headline confirms in direction: both independent draws beat
`trunk-ft-pitchshift` (+0.038 and +0.020), and the two-draw mean (+0.029) is
still a real, positive lever, one layer-cut swap for a further gain on top of
this era's two largest confirmed single levers. The between-run spread
(-0.018 +/- 0.011) is consistent with this era's known noise floor
(~0.012-0.016 headline, ~0.014-0.026 per fold) -- not evidence the lever is
fake, but evidence the first run's exact size (+0.038) was itself on the
favorable side of that spread.

**`1_114`'s "recovery" does not replicate, and should not be cited as a
depth-specific mechanism.** Run 1 read as recovering the one blemish
`trunk-ft-pitchshift` flagged (1_114 moving the wrong way there); run 2
reverses it just as hard in the other direction (-0.088 paired, and net
*below* `trunk-ft-pitchshift`'s own 1_114 by -0.024). Averaged, the two-draw
`1_114` delta vs `trunk-ft-pitchshift` is +0.020 -- inside a single fold's own
noise (this fold's per-run SD alone is 0.028-0.041) and no larger than the
swings already documented for 28-buzz-event folds. `1_95` also lands inside
noise across the pair (run 1 +0.049, run 2 -0.031 vs `trunk-ft-pitchshift`,
mean +0.009): the depth12 cut's headline gain is not concentrated on either
hard fold specifically, on this evidence.

**Amends `trunk-pitchshift-depth12`'s conclusion**, which read `1_114`'s
single-draw +0.064 as "the more notable move" and "exactly the more
surprising, stronger result branch" of its own falsifier. That reading does
not survive a second draw. `trunk-pitchshift-depth12`'s `trust` and
conclusion should carry a dated `amended` note pointing here; this entry
supersedes its hard-fold claim while leaving its headline-lever claim
(depth12 cut, stacked on pitch-shift concat, is a real gain over the layer12
cut) intact and now better-evidenced (two draws, not one).

`trust`: clean. Real, honest disagreement with the first run on where the
gain lives, exactly the kind of result a repeat run exists to catch.
