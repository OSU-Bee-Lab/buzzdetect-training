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
## Conclusion
