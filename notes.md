# fast32h16-a0375-c
## Hypothesis
The fast end of the clean rung-C frontier has fast32h16 a0.50 (2.51x, 0.650 at 56k) and
fast32h16 a0.25 (2.75x, 0.612 at 56k); speed saturates near 2.75x (front-end bound;
fast32h32 a0.25 was no faster). a0.375 (contaminated rung B: 2.61x, 0.576 vs a0.50 0.610 and
a0.25 0.533 on the same pack) sits midway in speed. Clean rung C WSD should put it ~0.63 at
~2.61x, above the straight line between its neighbours, which would make it a frontier point.
Matched controls: `fe_C_fast32h16_a0.50_s1_c-buzz-rain-human_wsd*` and
`fe_C_fast32h16_a0.25_s1_c-buzz-rain-human_wsd*` (width only).
Falsifier: at or below 0.612 it is dominated by a0.25 (faster, same headline).
## Changes
None to code. `05_distill/main.py --rung C --runs "fast32h16:a0.375:classes=ins_buzz+ambient_rain+human" --wsd-max 56000 --wsd-stop 1.3`.
