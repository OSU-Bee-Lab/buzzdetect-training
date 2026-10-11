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
