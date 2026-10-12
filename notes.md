# twofast32-d8-s2
## Hypothesis
Distillation arm (IDEAS D3), the repeat the rule owes. `twofast32-d8` (2026-10-11, seed 1) put two students
above the line by more than 0.02: `twofast32:a0.50_d8` at 0.701 / 5.33x, +0.033 over everything at its speed
or faster (`fast32h16:a0.75_d8` 0.656 at 5.57x), and `twofast32:a0.25_d8` at 0.668 / 6.14x, +0.022 over
`fast32h16:a0.50_d8` (0.646 / 0.642 at 6.7x). Neither counts until a second seed agrees.

This run repeats both at seed 2: same teacher, rung C, front end, arch, classes (buzz + rain + human) and
budgets (WSD 7000 / 14000 / 28000 / 56000, judged at 56000, `mmt` export). Nothing else changes; the
matched control of each student is its own seed 1.

Prediction: `a0.50_d8` holds (seed 2 within seed noise, 0.01-0.02, of 0.701, so still over the 0.676 a gain
of 0.02 needs). `a0.25_d8` is a coin flip: its seed-1 margin over the 0.02 bar is 0.002.
Falsifier (from IDEAS D3): seed 2 of `a0.50_d8` at or under 0.676, or `a0.25_d8` under 0.666.
Also read: the last doubling of the seed-2 `a0.50_d8` curve (seed 1 was still climbing, 0.687 -> 0.701).

## Changes
None to the method. `05_distill/` carries `exp/twofast32-d8`'s code as a patch (the `mmt` front end and
`reexport.py` are not in main). `chain.sh`: `main.py --rung C --seed 2 --wsd 7000,14000,28000,56000` on
both students, then `reexport.py --fe-impl mmt --suffix _mmt` on the new seed-2 branches. `results.sh`:
the four `tools/results.py` tables.

## Results
Seed 2 against seed 1 and against the bar each seed-1 gain was read over (headline at WSD 56000; speed is
the `_mmt` row's x YAMNet at 200 s on the GPU):

| student | seed 1 | seed 2 | seed 2 - seed 1 | bar (falsifier) | fastest-or-faster rival | seed 2 over rival |
|---|---|---|---|---|---|---|
| `twofast32:a0.50_d8` | 0.701 / 5.33x | 0.694 / 5.34x | -0.007 ± 0.005 | > 0.676 | `fast32h16:a0.75_d8` 0.656 / 5.57x | +0.038 ± 0.012 |
| `twofast32:a0.25_d8` | 0.668 / 6.14x | 0.668 / 5.85x | +0.000 ± 0.006 | >= 0.666 | `fast32h16:a0.50_d8` 0.646, 0.642 / 6.7x | +0.022 ± 0.011 |

(± SD is eval sampling only; headline seed noise on a student is 0.01-0.02.)

Step-budget curves, seed 2 (headline at 7000 / 14000 / 28000 / 56000):
- `a0.50_d8`: 0.675 / 0.677 / 0.691 / 0.694; hit@K +1.17, +0.81, +1.04. Seed 1: 0.663 / 0.673 / 0.687 / 0.701.
- `a0.25_d8`: 0.639 / 0.648 / 0.663 / 0.668; hit@K +2.09, +2.40, +0.86. Seed 1: 0.644 / 0.658 / 0.669 / 0.668.

`tools/results.py` tables (results.log):


### a0.50_d8: seed 1 -> seed 2

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.807 | 0.802 | -0.005 | 0.012 | 34 |
| 53 | 0.674 | 0.658 | -0.016 | 0.011 | 32 |
| 1_11 | 0.628 | 0.596 | -0.032 | 0.014 | 32 |
| 1_143 | 0.673 | 0.673 | +0.000 | 0.012 | 26 |
| 1_37 | 0.725 | 0.740 | +0.015 | 0.011 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.701 → this 0.694 (-0.007 ± 0.005)
- inclusive (sensitivity), same thresholds: 0.590 → 0.581 (-0.009)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.810 | 0.822 | +0.012 | 603 |
| untagged | 0.705 | 0.697 | -0.008 | 7485 |
| background | 0.701 | 0.690 | -0.011 | 9412 |
| quiet | 0.233 | 0.224 | -0.009 | 2231 |
| faint | 0.000 | 0.000 | +0.000 | 35 |


### a0.25_d8: seed 1 -> seed 2

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.823 | 0.792 | -0.031 | 0.012 | 34 |
| 53 | 0.681 | 0.666 | -0.015 | 0.012 | 32 |
| 1_11 | 0.565 | 0.597 | +0.032 | 0.015 | 32 |
| 1_143 | 0.649 | 0.633 | -0.016 | 0.019 | 26 |
| 1_37 | 0.623 | 0.654 | +0.031 | 0.013 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.668 → this 0.668 (+0.000 ± 0.006)
- inclusive (sensitivity), same thresholds: 0.562 → 0.559 (-0.003)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.792 | 0.788 | -0.004 | 603 |
| untagged | 0.665 | 0.666 | +0.001 | 7485 |
| background | 0.721 | 0.696 | -0.025 | 9412 |
| quiet | 0.208 | 0.208 | +0.000 | 2231 |
| faint | 0.000 | 0.000 | +0.000 | 35 |


### fast32h16:a0.75_d8 (s1) -> twofast32:a0.50_d8 s2

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.751 | 0.802 | +0.051 | 0.025 | 34 |
| 53 | 0.640 | 0.658 | +0.018 | 0.022 | 32 |
| 1_11 | 0.630 | 0.596 | -0.034 | 0.033 | 32 |
| 1_143 | 0.621 | 0.673 | +0.052 | 0.023 | 26 |
| 1_37 | 0.637 | 0.740 | +0.103 | 0.029 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.656 → this 0.694 (+0.038 ± 0.012)
- inclusive (sensitivity), same thresholds: 0.544 → 0.581 (+0.037)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.793 | 0.822 | +0.029 | 603 |
| untagged | 0.656 | 0.697 | +0.041 | 7485 |
| background | 0.654 | 0.690 | +0.036 | 9412 |
| quiet | 0.194 | 0.224 | +0.030 | 2231 |
| faint | 0.000 | 0.000 | +0.000 | 35 |


### fast32h16:a0.50_d8 (s1) -> twofast32:a0.25_d8 s2

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.728 | 0.792 | +0.064 | 0.023 | 34 |
| 53 | 0.647 | 0.666 | +0.019 | 0.024 | 32 |
| 1_11 | 0.608 | 0.597 | -0.011 | 0.021 | 32 |
| 1_143 | 0.607 | 0.633 | +0.026 | 0.024 | 26 |
| 1_37 | 0.642 | 0.654 | +0.012 | 0.032 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.646 → this 0.668 (+0.022 ± 0.011)
- inclusive (sensitivity), same thresholds: 0.536 → 0.559 (+0.023)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.778 | 0.788 | +0.010 | 603 |
| untagged | 0.649 | 0.666 | +0.017 | 7485 |
| background | 0.642 | 0.696 | +0.054 | 9412 |
| quiet | 0.193 | 0.208 | +0.015 | 2231 |
| faint | 0.000 | 0.000 | +0.000 | 35 |


Interpretation.
- **`a0.50_d8` repeats.** Seed 2 is 0.694, 0.007 under seed 1 and inside seed noise, and well over the
  0.676 falsifier. Against `fast32h16:a0.75_d8` (the best student at its speed or faster) it is
  +0.038 ± 0.012; the two-seed mean is 0.698 at 5.33x. Folds against that rival: `1_37` +0.103 ± 0.029,
  `1_143` +0.052 ± 0.023 and `1_29` +0.051 ± 0.025 are gains; `53` +0.018 ± 0.022 unsure; `1_11`
  -0.034 ± 0.033 unsure (about one SD down, and `1_11` is also where the two seeds disagree most,
  0.628 vs 0.596). Every tier is up: loud +0.029, untagged +0.041, background +0.036, quiet +0.030.
- **`a0.25_d8` repeats, at the margin it had.** Seed 2 is 0.668, identical to seed 1, so the falsifier
  (under 0.666) is not met. Its gain over `fast32h16:a0.50_d8` is +0.022 ± 0.011 (two-seed means 0.668
  vs 0.644, +0.024): just over the 0.02 bar and about two eval SDs, so a small real step, not a large
  one. The gain is mostly `background` (+0.054) and so mostly `1_29` (+0.064 ± 0.023); untagged +0.017,
  loud +0.010, quiet +0.015 are small; `53`, `1_11`, `1_143`, `1_37` are all within their SDs (unsure).
  Per-fold seed-to-seed swings on this student are +-0.03 (`1_29` -0.031, `1_11` +0.032, `1_37` +0.031)
  with the headline unchanged: fold-level readings on one seed of it are not safe.
- **Speed.** `a0.25_d8` seed 2 timed 5.85x against seed 1's 6.14x (same graph shape, different weights):
  a 5% spread between two timings of one architecture. It is still faster than `fast32h16:a0.75_d8`
  (5.57x) but the 6.14x figure should be read as 5.8-6.1x. `a0.50_d8` timed 5.33x and 5.34x.
- **Last doubling.** Seed 1's 0.687 -> 0.701 climb from 28000 to 56000 did not repeat: seed 2 went
  0.691 -> 0.694, and its hit@K gains (+0.81, +1.04) are under the 1.3 stop bar twice in a row. No case
  for a longer budget on this student.

## Conclusion
Both `twofast32` d8 gains count under the repeat rule. `twofast32:a0.50_d8` is on the frontier at
0.694-0.701 / 5.33x (+0.038 ± 0.012 over `fast32h16:a0.75_d8`, all tiers up, three of five folds clearly
up, `1_11` unsure-down). `twofast32:a0.25_d8` is on it at 0.668 / 5.85-6.14x (+0.022 ± 0.011 over
`fast32h16:a0.50_d8`), a small gain carried mostly by `background` / `1_29`. The curve is flat past 28000
on seed 2, so 56000 is enough. Trust: clean.
