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
