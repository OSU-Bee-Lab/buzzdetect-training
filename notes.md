# wsd112-yam64h32
## Hypothesis
The frontier student yam64h32:a0.25_d8 (rung C, buzz+rain+human, seed 1) gained +0.003 from 28000 to 56000 steps
(0.655 -> 0.658) and the WSD stop rule never fired, so the question of whether its step budget is exhausted is open.
Doubling the same trunk to 112000 steps (resumes the 56000 trunk; same data, seed, front end) tests it.
Prediction: 0.665 +- 0.015 at 112000 (a gain <= 0.01); speed unchanged (~6.8x). Falsifier: >= 0.68 (budget was
limiting; seed 2 reached 0.676 at 56000 so seed noise is the competing explanation, hence compare to s1 control 0.658).
Matched control: fe_C_yam64h32_a0.25_d8_s1_c-buzz-rain-human_wsd56000 (_mmt row).
## Changes
None to code. `main.py --rung C --wsd 7000,14000,28000,56000,112000` extends the existing trunk; `reexport.py --fe-impl mmt`.
## Results
Chain exited 0 (main.py WSD extension to 112000, reexport mmt, bench, record).
| run (s1, mmt) | steps | headline | hit@K | dK | x YAMNet (GPU 200 s) |
|---|---|---|---|---|---|
| wsd56000_mmt (control) | 56000 | 0.658 | 62.47 | +0.59 | 6.81 |
| wsd112000_mmt | 112000 | 0.662 | 63.25 | +0.78 | 7.68 |

Per fold 112k vs 56k: +0.009, -0.006, +0.004, +0.011, +0.003; mean +0.004 +- 0.003 (SE over folds).
Tiers: quiet 0.192 -> 0.197, background 0.658 -> 0.659, loud 0.806 -> 0.818.
Speed: identical architecture, so 6.81 -> 7.68x is bench noise (the same model read 4.31x on the 20 s bench in this run); treat speed as unchanged.
Seed 2 at 56000 is 0.676, so the seed-to-seed spread (0.018) is about 4x the 56k -> 112k gain.
## Conclusion
Prediction held (0.662, inside 0.665 +- 0.015) and the falsifier (>= 0.68) was not met. Doubling the budget again gives
+0.004 +- 0.003. hit@K gained <1.3 on 28k->56k and 56k->112k, so the WSD stop rule (two doublings below 1.3) agrees.
yam64h32:a0.25_d8's step budget is exhausted at about 56000. The remaining gap to seed 2 is seed variance, not
training length. Implication: further gains for this student have to come from architecture, front end or teacher;
don't extend WSD past 56000 for a0.25-sized students.
