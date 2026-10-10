# fast32-d8-repeat
## Hypothesis
Distillation arm, repeat rule. `fe_C_fast32_a0.50_d8_s1_c-buzz-rain-human_wsd56000` (0.685 at 2.34x YAMNet) is +0.021 over every rung-C student at its speed or faster, on one seed; `ladder_record.py frontier` lists it as awaiting a repeat. Headline seed noise on a student is ~0.01-0.02, so the gain counts only if seed 2 of the identical run (same teacher, rung, WSD budgets, front end, width, depth, classes) lands within that noise of seed 1 and still above everything at its speed or faster. Falsified if seed 2 falls back onto the existing frontier (gain <= 0.02 over `fast32 a0.50_d6`, 0.664 at 2.38x).

Matched control: the seed-1 row itself, at the same budget.

## Changes
None. `ladder_record.py frontier`'s repeat command, verbatim:
`05_distill/main.py --rung C --seed 2 --wsd 7000,14000,28000,56000 --runs "fast32:a0.50_d8:classes=ins_buzz+ambient_rain+human"`
