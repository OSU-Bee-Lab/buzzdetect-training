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
