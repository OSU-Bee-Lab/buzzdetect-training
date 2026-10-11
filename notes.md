# fast32h16-a025-repeat
## Hypothesis
Distillation arm, repeat rule. `fe_C_fast32h16_a0.25_s1_c-buzz-rain-human_wsd56000` (0.612 at 2.75x YAMNet)
is the fastest rung-C student, so `ladder_record.py frontier` lists it as awaiting a repeat: nothing at its
speed or faster exists, and its whole headline is the "gain". It is also the control `fused-blocks` judged
`a0.25_d8_f2` against (0.630 at 2.73x, +0.018 +- 0.008 on one seed each), so the fast end of the frontier
rests on single seeds. Headline seed noise on a student is ~0.01-0.02: the point counts if seed 2 of the
identical run (same teacher, rung, WSD budgets, front end, width, classes) lands within that noise of
seed 1. Falsified if seed 2 differs from seed 1 by more than 0.02 at the judged budget (56000).

Matched control: the seed-1 row itself, at the same budget.

## Changes
None. `ladder_record.py frontier`'s repeat command, verbatim:
`05_distill/main.py --rung C --seed 2 --wsd 7000,14000,28000,56000 --runs "fast32h16:a0.25:classes=ins_buzz+ambient_rain+human"`
