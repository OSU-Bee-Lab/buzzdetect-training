# yam64h32-s2
## Hypothesis
Distillation arm, the repeat the rule owes (`ladder_record.py repeats`). `yam64h32` (2026-10-14, seed 1) put
`yam64h32:a0.25_d8` at 0.658 / 6.81x (WSD 56000, mmt), more than 0.02 over everything at its speed or faster
(`fast32h16:a0.25_d8_f2` 0.630 / 6.83x, `fast32h16:a0.25` 0.612-0.616 / 7.3x). It doesn't count until a
second seed agrees.

This run repeats it at seed 2: same teacher, rung C, front end, arch, classes (buzz + rain + human), budgets
(WSD 7000 / 14000 / 28000 / 56000, judged at 56000, `mmt` export). The matched control is seed 1.

Prediction: seed 2 within seed noise (0.01-0.02) of 0.658, so still above 0.650 (the 0.02 bar over 0.630).
Falsifier: seed 2 at or under 0.650.

## Changes
None to the method. `05_distill/` carries `exp/yam64h32`'s code as a patch (`mmt` front end, `reexport.py`,
`yam64h32` registered). `chain.sh`: `main.py --rung C --seed 2 --wsd 7000,14000,28000,56000` on the student,
then `reexport.py --fe-impl mmt --suffix _mmt` on its seed-2 branches.

## Results

## Conclusion
