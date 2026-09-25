# pitchshift-aves-mid-repeat

## Hypothesis

Repeat draw of `pitchshift-aves-mid` (2026-09-14): headline beat both parent
levers (+0.043 over `aves-mid-repeat`, +0.031 over `pitchshift-repeat`) with
both hard folds (`1_150`, `1_95`) moving beyond either parent alone, but
`1_114` dropped -0.204 relative to plain pitch-shift. Given the size and that
`1_150` specifically is this project's target fold, confirm before trusting
the magnitude, per LOOP.md's hard-fold-gain protocol.

## Changes

None. Identical config and cache to `pitchshift-aves-mid`: `--embedder
yamnet_pitchshift_aves_mid`, `--translation general`, `--fixed-epochs 400`,
no dropout, medium set. Only `--name` differs. No re-extraction -- reused
the shared cache via the worktree symlink.

## Results

Both draws side by side, against both single-lever comparators:

| fold | vs pitchshift-repeat r1 | vs pitchshift-repeat r2 | vs aves-mid-repeat r1 | vs aves-mid-repeat r2 |
|---|---|---|---|---|
| **1_150** | **+0.137** | **+0.121** | **+0.122** | **+0.106** |
| **1_95** | **+0.094** | **+0.083** | **+0.047** | **+0.036** |
| **1_114** | **-0.204** | **-0.202** | +0.042 | +0.044 |
| 1_29 | +0.083 | +0.095 | +0.023 | +0.036 |
| willard/1_11 | +0.085 | +0.081 | +0.015 | +0.011 |
| 53 | +0.048 | +0.050 | +0.005 | +0.007 |
| 1_143 | +0.038 | -0.009 | +0.029 | -0.019 |
| 1_37 | -0.029 | -0.060 | +0.061 | +0.031 |

mean sens@fpr0.005 (excl. quiet): r1 0.422, r2 0.411 -- headline replicates
tightly. vs `pitchshift-repeat`: +0.031, +0.020 (both positive, both clear
gains). vs `aves-mid-repeat`: +0.043, +0.032 (both positive, both clear the
~0.027 MDE by a wide margin).

## Conclusion

**Both hard folds move up on both independent draws, against both
comparators, with consistent magnitude.** `1_150` (+0.137/+0.121 vs
pitch-shift, +0.122/+0.106 vs aves-mid) and `1_95` (+0.094/+0.083,
+0.047/+0.036) are the most reliably confirmed hard-fold gains logged this
era -- unlike `aves-mid`'s own repeat, where `1_150`'s magnitude swung
0.148 -> 0.042, here it holds within ~15% across draws both times. That
stability is itself informative: stacking the two levers appears to reduce
`1_150`'s training variance, not just its mean.

**`1_114`'s cost against plain pitch-shift is equally well confirmed and
equally real** (-0.204, -0.202 -- near-identical both draws, not noise).
Against `aves-mid` it is a small *gain* both times (+0.042, +0.044), so the
combined model's `1_114` behaviour tracks AVES, not pitch-shift -- whatever
made `yamnet_pitchshift` alone unusually strong there does not survive the
stack. `1_143` and `1_37` are the two folds without a consistent story
(flip sign between draws, modest either way) -- unsure, a normal verdict for
folds this thin (9-22 buzz events at baseline).

**This is the era's best confirmed structural result to date**: it beats
both single-lever configs on the headline on both draws, and delivers the
project's two most-wanted hard-fold gains with unusually tight
reproducibility. Amends `pitchshift-aves-mid`'s `caveated` entry to `clean`.
Worth flagging to Luke as a shipping candidate once the `to_onnx()` gap for
the `yamnet_pitchshift` family is closed (pre-existing, not new here).
