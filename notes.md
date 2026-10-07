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

## Results
Rung C, WSD, seed 1, eval on the 5 rotating folds (headline = sensitivity_exclquiet at matched FPR).

| steps | a0.375 | a0.50 control | a0.25 control | a0.375 hit@K | a0.375 x_yamnet200 |
|---|---|---|---|---|---|
| 7k  | 0.592 | 0.605 | 0.546 | 55.8 | 2.55 |
| 14k | 0.611 | 0.621 | 0.569 | 59.7 | 2.61 |
| 28k | 0.626 | 0.639 | 0.603 | 62.1 | 2.60 |
| 56k | 0.639 | 0.650 | 0.612 | 62.4 | 2.63 |

Control speed (x_yamnet200): a0.50 ~2.50, a0.25 ~2.73. results.py at 56k: vs a0.50 -0.011 ± 0.007,
vs a0.25 +0.027 ± 0.007 (eval sampling only). Per-loudness-tier sensitivity vs a0.50: loud -0.036, background -0.017,
untagged -0.007, quiet -0.001. Per fold vs a0.50: 1_29 -0.032, 53 +0.008, 1_11 +0.031, 1_143 -0.038, 1_37 -0.024
(SD 0.012-0.019 each; single seed, so weak). Stop rule: hit@K gains +3.86, +2.40, +0.32, so only one small gain and it ran to 56k.
The a0.50 seed spread at 28k (s1 0.639, s2 0.650) gives a feel for training noise, about 0.01.

## Conclusion
Falsifier not met: 0.639 at 56k beats a0.25's 0.612 by +0.027 ± 0.007 and the prediction (~0.63 at ~2.61x) held.
It is not dominated, but it sits on the straight line between its neighbours, not above it: interpolating
a0.50 (2.50x, 0.650) to a0.25 (2.73x, 0.612) at 2.6x gives ~0.633, and at 28k ~0.624 vs a measured 0.626. Both gaps are
well inside one seed's noise. So width trades headline for speed about linearly across a0.25-a0.50 on fast32h16
(~-0.04 headline per +0.1x). a0.375 is a usable intermediate frontier point, not a knee. Speed itself saturates
toward 2.75x (front-end bound), so going narrower on this front end buys little speed.
