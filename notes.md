# yamnet-pitchshift-x4

## Hypothesis

IDEAS.md item 5's untested remainder: if the x2 rung (pitch-shift up one
octave, concat with unshifted YAMNet, +0.069 headline this era) is
directionally positive, a third block from a second octave up (resample
16k->4k, tile 4x) concatenated onto the existing 2048-d cache might extend
the gain further. Falsifier: if x4's own hard-fold pattern doesn't track x2's
(particularly `1_114`, x2's standout fold), the win may be about *a* shift
rather than specifically the octave chosen.

## Changes

New `embedders/yamnet_pitchshift_x4/embedder.py`, subclassing
`EmbedderYamnetPitchshift` and adding a third 1024-d block (resample
16k->4k, tile 4x -- harmonics at 920/1840/2760 Hz instead of x2's
460/920/1380 Hz), giving 3072-d total. Same frame grid, same
`--fixed-epochs 400`, bare linear probe, `--translation general` as the x2
control.

## Results

| fold | x2 (base) | x4 (this) | delta | delta SD (eval sampling) | buzz events |
|---|---|---|---|---|---|
| 1_150 | 0.352 | 0.303 | -0.049 | 0.023 | 21 |
| willard | 0.446 | 0.420 | -0.026 | 0.022 | 26 |
| 1_114 | 0.431 | 0.408 | -0.023 | 0.023 | 28 |
| 1_29 | 0.479 | 0.469 | -0.010 | 0.013 | 32 |
| 1_143 | 0.541 | 0.532 | -0.009 | 0.019 | 22 |
| 53 | 0.433 | 0.425 | -0.008 | 0.013 | 28 |
| 1_37 | 0.430 | 0.444 | +0.014 | 0.030 | 14 |
| 1_95 | 0.076 | 0.097 | +0.021 | 0.014 | 46 |

mean sens@fpr0.005: x2 0.398 -> x4 0.387 (-0.011 +/- 0.007 headline delta SD,
eval sampling only). 6/8 folds down, 2 up. Tiers are uniformly flat-to-down
(loud 0.837->0.834, untagged 0.430->0.419, background 0.326->0.316, quiet
0.088->0.081) -- no tier-specific mechanism, just a small broad cost.

## Conclusion

**Falsifier fires.** `1_114` (x2's standout fold, +0.181 over the plain-YAMNet
anchor) goes the wrong way relative to x2 here (-0.023), and `1_150` is the
largest single mover, also down. The third block does not extend the gain --
if anything it costs a little, though the headline delta (-0.011 +/- 0.007)
is small enough to call this "no further gain" rather than a confident
negative. x2 remains the best rung; a second octave of shift does not help.
Settles the x4 remainder of IDEAS.md item 5. Shipped model not trained
(--skip-cv not run) since x4 isn't the winner.
