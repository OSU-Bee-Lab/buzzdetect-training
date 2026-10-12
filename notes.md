# twofast32-d8
## Hypothesis
Distillation arm. `matmul-stft` (2026-10-11) took the waveform Pad out of the exported graph, and the
frontier re-ordered: `twofast32 a0.50` (two channels at 60 frames: a 32 ms window on 80-1500 Hz, a 16 ms
window on 1500-7800 Hz) is 0.691 at 4.81x, +0.035 to +0.041 over every `fast32h16` student (0.612-0.656 at
5.6-7.3x) and level with the `fast32` students at their speed. It is the only front end whose trunk was
never cut: one student, full depth, a0.50. Cutting the trunk to 8 layers cost nothing on the other three
front ends (yamnet a0.50 0.725 -> d8 0.735 / 0.725; fast32 a0.50 0.687 -> d8 0.685 / 0.678; fast32h16 a0.50
0.650 -> d8 0.646 / 0.642), and without the Pad a trunk cut is worth real speed (fast32h16 a0.50 5.7x -> d8
6.7x).

Prediction: `twofast32:a0.50_d8` holds its full-depth twin's headline within seed noise (>= 0.671) and
times faster than 5.57x, i.e. it sits more than 0.02 above everything at its speed or faster (the best
there is fast32h16 a0.75_d8 0.656 at 5.57x, a0.50_d8 0.646 at 6.7x): a new frontier point, to be repeated
at seed 2 if so. `twofast32:a0.25_d8` asks how far the cut goes: a gain if it is > 0.02 above fast32h16
a0.50_d8 (0.646 / 0.642) at >= 6.7x, or above a0.25 (0.612 / 0.616) at >= 7.3x.
Falsified if the d8 cut costs twofast32 more than 0.02 (its low channel's 32 ms window needs the depth),
or if the two-channel front end keeps the student under 5.6x whatever the trunk (front end alone: 6.9x).

Matched control: `fe_C_twofast32_a0.50_s1_c-buzz-rain-human_wsd56000` (same front end, width, classes,
rung, WSD schedule, seed; full depth), its `_mmt` row for speed.

### Speed bench before training (random weights, pad-free `mmt` export, 200 s, GPU; `bench.log`)
| trunk on twofast32 | x YAMNet | MMACs |
|---|---|---|
| a0.50 (the control's trunk) | 4.80 | 5.9 |
| a0.50_d10 | 5.15 | 4.0 |
| **a0.50_d8** | **5.36** | 3.0 |
| a0.75_d8 | 4.68 | 6.3 |
| a0.375_d8 | 5.76 | 1.7 |
| **a0.25_d8** | **6.20** | 0.8 |
| a0.25 | 5.87 | 1.6 |
| front end alone | 6.55 | |

Same session: fast32h16 a0.75_d8 5.58x, a0.50_d8 6.72x, a0.25 7.34x.

**Revised before training.** The speed half of the prediction is already short: `a0.50_d8` times 5.36x, not
above 5.57x (the two-channel front end alone is 6.55x, so no trunk passes ~6.2x on it). The frontier test
stands as the rule states it: at 5.36x the best student at its speed or faster is still fast32h16
a0.75_d8 (0.656), so `twofast32:a0.50_d8` is a new point if it scores above 0.676, and `a0.25_d8` (6.20x;
best at its speed or faster: fast32h16 a0.50_d8 0.646 / 0.642 at 6.7x) if above 0.666.

## Changes
No code change. `chain.sh` distils `twofast32:a0.50_d8` and `twofast32:a0.25_d8` (rung C, seed 1, classes
buzz+rain+human, WSD 7000-56000 with `--wsd-stop 1.3`; neither trunk was stopped early), then re-exports
each judged branch with the pad-free front end (`reexport.py --fe-impl mmt --suffix _mmt`, same weights).
`bench.log` is the random-weight speed bench above; `results.sh` / `results.log` the tables below.
Headlines are identical in the plain and `_mmt` rows; speeds are the `_mmt` rows'.

## Results
Step-budget curves (headline at each budget; x YAMNet at 200 s, GPU, `_mmt`):

| student | 7000 | 14000 | 28000 | 56000 | speed |
|---|---|---|---|---|---|
| twofast32 a0.50_d8 | 0.663 | 0.673 | 0.687 | **0.701** | 5.33x |
| twofast32 a0.25_d8 | 0.644 | 0.658 | 0.669 | **0.668** | 6.14x |
| control twofast32 a0.50 (full depth) | 0.659 | 0.678 | 0.690 | 0.691 | 4.81x |

### a0.50_d8 against the matched control (twofast32 a0.50, 0.691 at 4.81x)
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.814 | 0.807 | -0.007 | 0.012 | 34 |
| 53 | 0.662 | 0.674 | +0.012 | 0.011 | 32 |
| 1_11 | 0.620 | 0.628 | +0.008 | 0.016 | 32 |
| 1_143 | 0.661 | 0.673 | +0.012 | 0.018 | 26 |
| 1_37 | 0.699 | 0.725 | +0.026 | 0.019 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.691 → this 0.701 (+0.010 ± 0.007)
- inclusive (sensitivity), same thresholds: 0.578 → 0.590 (+0.012)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.800 | 0.810 | +0.010 | 603 |
| untagged | 0.692 | 0.705 | +0.013 | 7485 |
| background | 0.703 | 0.701 | -0.002 | 9412 |
| quiet | 0.215 | 0.233 | +0.018 | 2231 |
| faint | 0.000 | 0.000 | +0.000 | 35 |

### a0.25_d8 against the matched control (twofast32 a0.50, 0.691 at 4.81x)
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.814 | 0.823 | +0.009 | 0.015 | 34 |
| 53 | 0.662 | 0.681 | +0.019 | 0.015 | 32 |
| 1_11 | 0.620 | 0.565 | -0.055 | 0.017 | 32 |
| 1_143 | 0.661 | 0.649 | -0.012 | 0.021 | 26 |
| 1_37 | 0.699 | 0.623 | -0.076 | 0.020 | 15 |

- mean: 0.691 → 0.668 (-0.023 ± 0.008); inclusive 0.578 → 0.562 (-0.016)
- tiers: loud -0.008, untagged -0.027, background +0.018, quiet -0.007

### Against the frontier students they displace
`a0.50_d8` (5.33x) against fast32h16 a0.75_d8 (0.656 at 5.57x), the best at its speed or faster before this run:

| fold | baseline | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.751 | 0.807 | +0.056 | 0.021 | 34 |
| 53 | 0.640 | 0.674 | +0.034 | 0.025 | 32 |
| 1_11 | 0.630 | 0.628 | -0.002 | 0.034 | 32 |
| 1_143 | 0.621 | 0.673 | +0.052 | 0.024 | 26 |
| 1_37 | 0.637 | 0.725 | +0.088 | 0.025 | 15 |

- mean: 0.656 → 0.701 (+0.045 ± 0.012); inclusive 0.544 → 0.590 (+0.046)
- tiers: loud +0.017, untagged +0.049, background +0.047, quiet +0.039

`a0.25_d8` (6.14x) against fast32h16 a0.50_d8 seed 1 (0.646 at 6.67x; seed 2 0.642 at 6.77x):

| fold | baseline | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.728 | 0.823 | +0.095 | 0.028 | 34 |
| 53 | 0.647 | 0.681 | +0.034 | 0.026 | 32 |
| 1_11 | 0.608 | 0.565 | -0.043 | 0.026 | 32 |
| 1_143 | 0.607 | 0.649 | +0.042 | 0.033 | 26 |
| 1_37 | 0.642 | 0.623 | -0.019 | 0.031 | 15 |

- mean: 0.646 → 0.668 (+0.022 ± 0.013); inclusive 0.536 → 0.562 (+0.026)
- tiers: loud +0.014, untagged +0.016, background +0.079, quiet +0.015

± SD is eval sampling only; student seed noise on a headline is another ~0.01-0.02 and is not in it.

Interpretation.
- **a0.50_d8: the d8 cut costs twofast32 nothing and buys 11% speed.** 0.701 against the full-depth twin's
  0.691 (+0.010 ± 0.007, inside seed noise: read it as "held", not as a gain from cutting), at 5.33x
  against 4.81x. No fold moves past 1.4 SD (`1_37` +0.026 ± 0.019, `53` +0.012 ± 0.011: unsure); every
  tier is within 0.02. The same headline faster is a frontier gain by the rule, and it dominates its own
  full-depth control. Against everything at its speed or faster it is +0.033 (frontier table; +0.045 ±
  0.012 over fast32h16 a0.75_d8, on four of five folds past 1.3 SD and every tier, `untagged` +0.049 and
  `background` +0.047), above the 0.676 bar set before training. That is over the 0.02 repeat threshold, so
  it owes a seed-2 repeat before it counts.
- **a0.50_d8 was still climbing at 56000** (0.687 → 0.701 on the last doubling; hit@K +0.82, which is
  under the stop rule's 1.3 but one doubling, so the trunk ran out its budget). The control had flattened
  (0.690 → 0.691). One draw: the last-doubling +0.014 is within seed noise.
- **a0.25_d8: a marginal pass.** 0.668 at 6.14x, 0.002 over its 0.666 bar: +0.022 ± 0.013 over fast32h16
  a0.50_d8 seed 1 and +0.026 over seed 2, but 8% slower than that student (6.14x against 6.7x), so it is
  a new point between it and a0.50_d8 rather than a replacement. The gain is `background` (+0.079, fold
  `1_29` +0.095 ± 0.028); `1_11` goes the other way (-0.043 ± 0.026). Against the twofast32 control the
  narrower trunk costs -0.023 ± 0.008, all of it on `1_11` (-0.055 ± 0.017) and `1_37` (-0.076 ± 0.020) and
  in `untagged` (-0.027), while `background` holds (+0.018): the width cut takes isolated buzz, not the
  long background ones. It flattened at 28000 (0.669 → 0.668). Also over the repeat threshold (+0.022),
  barely.
- Speed half of the original prediction was already withdrawn before training (bench 5.36x and 6.20x;
  trained 5.33x and 6.14x, so the random-weight bench predicts the trained speed within 1%).

## Conclusion
Cutting twofast32's trunk to 8 layers holds its headline, as on the three other front ends: `a0.50_d8`
is 0.701 at 5.33x against the full-depth 0.691 at 4.81x (+0.010 ± 0.007, held; all tiers within 0.02),
and is the new frontier point between yamnet a0.25_d8 (0.713 at 3.24x) and the fast32h16 students: +0.033
over everything at its speed or faster (fast32h16 a0.75_d8 0.656 at 5.57x; untagged +0.049, background
+0.047, loud +0.017). `a0.25_d8` is 0.668 at 6.14x, +0.022 over fast32h16 a0.50_d8 (0.646 at 6.67x) but
slower than it, gain in `background` (+0.079) with `1_11` down. Both are single seeds over the 0.02
repeat threshold: a seed-2 repeat of each is queued in IDEAS.md (D3) and neither counts until it lands.
The fast end beyond 6.2x still belongs to fast32h16 (the two-channel front end alone is 6.55x).
