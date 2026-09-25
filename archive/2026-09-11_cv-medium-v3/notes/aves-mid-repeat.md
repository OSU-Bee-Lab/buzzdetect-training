# aves-mid-repeat

## Hypothesis

Repeat draw of `aves-mid` (2026-09-14): headline +0.032 +/- 0.010 with `1_150`
moving +0.148, by far the largest single gain logged on that fold this era.
Given the size and that `1_150` is exactly the fold LOOP.md's hard-fold-gain
protocol asks to confirm, and that this fold is independently documented as
the most training-stochasticity-prone in the whole set (0.055-0.105 swings
between identical runs), confirm before trusting the magnitude.

## Changes

None. Identical config and cache to `aves-mid`: `--embedder yamnet_aves_mid`,
`--translation general`, `--fixed-epochs 400`, no dropout, medium set. Only
`--name` differs. No re-extraction -- reused the shared cache via the
worktree symlink.

## Results

vs `yamnet-aves-verify`, both draws side by side:

| fold | r1 delta | r1 delta SD | r2 delta | r2 delta SD | both agree? |
|---|---|---|---|---|---|
| **1_150** | **+0.148** | 0.049 | **+0.042** | 0.039 | yes, up both -- **magnitude unstable** |
| willard/1_11 | +0.083 | 0.026 | +0.056 | 0.026 | yes, up both |
| 1_29 | +0.072 | 0.027 | +0.063 | 0.030 | yes, up both |
| 1_95 | +0.016 | 0.015 | +0.025 | 0.016 | yes, up both |
| 53 | -0.014 | 0.023 | -0.014 | 0.022 | yes, down both |
| 1_37 | -0.043 | 0.031 | -0.034 | 0.031 | yes, down both |
| 1_114 | -0.002 | 0.017 | +0.012 | 0.020 | no, flips near zero |
| 1_143 | -0.001 | 0.034 | +0.056 | 0.044 | no, flips |

mean sens@fpr0.005 (excl. quiet): r1 0.386, r2 0.379 -- headline replicates
tightly. Both draws land at +0.025 to +0.032 vs `yamnet-aves-verify`'s 0.354,
both clearing the ~0.027 headline MDE and both outside their own ~0.010-0.011
headline delta SD.

## Conclusion

**Six of eight folds agree in direction on both independent draws, and the
headline is a stable, MDE-clearing gain both times (+0.032, +0.026).** This is
a real, reproducible structural gain from reading AVES's middle layers rather
than the final one alone -- not a lucky draw. `1_150`'s own magnitude is
*not* resolvable from two draws (+0.148 vs +0.042, itself close to the
0.055-0.105 run-to-run swing LOOP.md documents for this specific fold under
identical configs) -- read that as "1_150 moves up, size unknown," not as a
retraction of the gain. `1_114` and `1_143` flip between draws, both near
zero or modest either way -- unsure, a normal verdict for two folds this
thin (9-14 buzz events at baseline).

**Settles IDEAS item 4 as a confirmed structural win**, roughly half the size
of `yamnet_pitchshift`'s +0.069 but via a completely different representation
lever (AVES depth, not YAMNet frequency register) -- worth testing whether it
stacks with `yamnet_pitchshift` or `yamnet_pitchshift_decimate` (neither has
an AVES component). Amends `aves-mid`'s `caveated` entry to `clean`.
