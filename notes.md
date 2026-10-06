# fast32-a0375-c
## Hypothesis
The clean rung-C frontier has a gap between fast32 a0.50 (2.17x, 0.680 at 28k) and
fast32h16 a0.50 (2.51x, 0.639/0.650 at 28k, two seeds). Contaminated rung-B readings put
fast32 a0.375 at 2.29x and 0.628 (vs fast32 a0.50's 0.648 at the same rung), i.e. -0.02
for +0.12x. Retrained clean on rung C with WSD, fast32 a0.375 should land ~0.66 at ~2.29x,
above fast32h16 a0.50 at a speed between the two: a new non-dominated point.
Matched control: `fe_C_fast32_a0.50_s1_c-buzz-rain-human_wsd*` (same front end, rung, schedule,
classes, seed; width only). Falsifier: headline at or below 0.650 (fast32h16 a0.50's best)
means width costs more than the front end and the point is dominated.
## Changes
None to code. `05_distill/main.py --rung C --runs "fast32:a0.375:classes=ins_buzz+ambient_rain+human" --wsd-max 56000 --wsd-stop 1.3`.

## Results
Rung C, WSD, seed 1, eval on the 5 rotating folds (headline = sensitivity_exclquiet at matched FPR).

| steps | a0.375 headline | a0.50 control | Δ | a0.375 hit@K | a0.375 x_yamnet200 |
|---|---|---|---|---|---|
| 7k  | 0.649 | 0.654 | -0.005 | 57.8 | 2.29 |
| 14k | 0.654 | 0.668 | -0.014 | 62.8 | 2.28 |
| 28k | 0.666 | 0.680 | -0.014 | 65.3 | 2.30 |
| 56k | 0.675 | 0.687 | -0.012 | 65.8 | 2.27 |

Control speed: x_yamnet200 2.15-2.17. results.py at 28k: -0.014 ± 0.006 (eval sampling), every fold negative-or-flat (1_143 -0.026 ± 0.012). Stop rule: hit@K gains +5.0, +2.5, +0.45; only one small gain, so it ran to 56k.
Per fold at 56k (a0.375 vs a0.50): 0.754/0.760, 0.658/0.658, 0.658/0.668, 0.611/0.660, 0.696/0.688. The
fourth fold carries most of the gap (-0.049); single seed, per-fold SD ~0.01-0.02, so weak.

## Conclusion
Falsifier not met: 0.666 at 28k / 0.675 at 56k sits above fast32h16 a0.50's 0.650, at ~2.28x, between
fast32 a0.50 (2.16x, 0.687) and fast32h16 a0.50 (2.51x). Width costs ~-0.012 to -0.014 headline for
+0.12x speed, steady across budgets, slightly better than the contaminated rung-B reading (-0.02).
fast32 a0.375 is a new non-dominated rung-C frontier point (single seed; the gap to a0.50 is about one
combined-headline SD, so treat its rank vs a0.50-at-lower-speed as a trade, not a loss).
