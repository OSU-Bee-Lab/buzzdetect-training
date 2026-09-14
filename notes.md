# pitchshift-repeat

## Hypothesis

`yamnet-pitchshift` (2026-09-13) is the era's largest clean gain (+0.069, 8/8
folds up) but rests on a single draw, same as every other run this era —
there is no seed control. A repeat draw of the identical config is the cheap
check LOOP.md prescribes before trusting a result of that size: does the
headline direction and magnitude hold, and do the two named hard folds move
consistently?

## Changes

None. Identical config to `exp/yamnet-pitchshift`: `--embedder
yamnet_pitchshift`, `--translation general`, `--fixed-epochs 400`, no
dropout, medium set. Only `--name` differs.

## Results

| fold | r1 sens@fpr0.005 | r2 (this run) | delta | delta SD (eval sampling) | buzz events |
|---|---|---|---|---|---|
| 1_150 | 0.352 | 0.331 | -0.021 | 0.021 | 21 |
| 1_95 | 0.076 | 0.076 | +0.000 | 0.012 | 46 |
| 1_29 | 0.479 | 0.468 | -0.011 | 0.008 | 32 |
| 53 | 0.433 | 0.436 | +0.003 | 0.008 | 28 |
| willard/1_11 | 0.446 | 0.430 | -0.016 | 0.013 | 26 |
| wooster/1_143 | 0.541 | 0.533 | -0.008 | 0.009 | 22 |
| 1_37 | 0.430 | 0.439 | +0.009 | 0.014 | 14 |
| 1_114 | 0.431 | 0.414 | -0.017 | 0.014 | 28 |

- mean sens@fpr0.005: r1 0.398 -> r2 0.391 (headline delta -0.007, delta SD
  0.005, eval sampling only)

2 folds up, 5 down, 1 flat — but every single fold delta is inside ~1-1.5x
its own eval-sampling delta SD, and the headline delta (-0.007 ± 0.005) is
an order of magnitude smaller than the +0.069 gain being checked. Both named
hard folds are unremarkable: 1_150 -0.021 ± 0.021 (~1σ, direction not even
resolvable), 1_95 dead flat (+0.000 ± 0.012). No fold failed to reach the
target FPR in either draw.

## Conclusion

**Confirms `yamnet-pitchshift`'s headline.** Two independent draws of the
same config land at 0.398 and 0.391 against a delta SD of 0.005 — the repeat
is not distinguishable from the original at the fold level or the headline
level, and neither draw's hard-fold pattern should be read as a mechanism
(same caution `hidden-aves-verify-r2` already established for 1_150 alone).
`yamnet_pitchshift` (+0.069 over `cv_baseline_v3`) stands as the era's
best-supported single lever. Nothing further to test here; the era-leader
verdict is settled, not open.
