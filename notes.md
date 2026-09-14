# pitchshift-down-repeat

## Hypothesis

`yamnet-pitchshift-down` (2026-09-13) was a small, broadly positive result
(+0.019 over `cv_baseline_v3`, 5/8 folds up, all tiers moving together) that
its own log entry flagged as "near the ~0.027 headline MDE at n=1... a repeat
draw would be the natural next step before trusting the 1_150 movement
specifically." This is that repeat draw: identical config, no code changes.

## Changes

None to the embedder or training config. `embedders/yamnet_pitchshift_down`
was copied into this worktree as a real (unsymlinked) directory — it lives
only on `exp/yamnet-pitchshift-down`, not yet promoted to the shared tree the
way `yamnet_pitchshift` was — since the medium set's embeddings cache for it
already exists in the shared tree, no re-extraction was needed.

## Results

| fold | r1 sens@fpr0.005 | r2 (this run) | delta | delta SD (eval sampling) | buzz events |
|---|---|---|---|---|---|
| 1_150 | 0.352 | 0.315 | -0.037 | 0.030 | 21 |
| 1_95 | 0.031 | 0.031 | +0.000 | 0.007 | 46 |
| 1_29 | 0.452 | 0.445 | -0.008 | 0.008 | 32 |
| 53 | 0.438 | 0.432 | -0.006 | 0.008 | 28 |
| willard/1_11 | 0.404 | 0.395 | -0.009 | 0.010 | 26 |
| wooster/1_143 | 0.450 | 0.450 | +0.000 | 0.009 | 22 |
| 1_37 | 0.424 | 0.425 | +0.002 | 0.011 | 14 |
| 1_114 | 0.241 | 0.225 | -0.016 | 0.020 | 28 |

- mean sens@fpr0.005: r1 0.349 -> r2 0.340 (headline delta -0.009, delta SD
  0.005, eval sampling only)
- r2 vs the era anchor (`cv_baseline_v3`, 0.330): +0.010, smaller than r1's
  +0.019 but the same sign.

1 fold up, 5 down, 2 flat between draws. Every fold delta is inside ~1-1.2x
its own eval-sampling delta SD, including 1_150 (-0.037 ± 0.030, ~1.2σ) — the
fold r1 called its standout mover. Direction is unresolved at the fold level;
the headline direction (small positive gain over the anchor) replicates.

## Conclusion

**Weak evidence holds up as weak evidence, nothing stronger.** Two draws of
`yamnet_pitchshift_down` land at +0.019 and +0.010 over `cv_baseline_v3` —
consistent in sign, both comfortably below the ~0.027 headline MDE, in line
with the original entry's own framing ("weak evidence for a real if modest
gain, not a confident win"). The 1_150 story specifically does not replicate
with any confidence: r1's +0.086-over-anchor read as a standout is not
distinguishable from r2's smaller movement given a ~0.03-0.04 combined
sampling+training SD on that fold (same lesson as `hidden-aves-verify-r2`).
Read `yamnet_pitchshift_down` as: probably a small real gain, unclear whether
it does anything for 1_150 specifically, and not worth further investment
relative to `yamnet_pitchshift` (x2 up, +0.069, now twice-confirmed).
