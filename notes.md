# yamnet-d8-repeat
## Hypothesis
Repeat rule (LOOP.md distillation baseline): `yamnet-depth`'s yamnet a0.50_d8 (0.735 @1.52x, seed 1) beats every student at its speed or faster by > 0.02 (a0.25 0.707 @1.62x, +0.028), so it gets one repeat at the next seed before its point counts. `ladder_record.py repeats` does not flag it because it only considers cosine runs. Student: yamnet a0.50_d8, seed 2, same spec. Rung C, buzz+rain+human (select), WSD to 56k with the stop rule (--wsd-max 56000 --wsd-stop 1.3), teacher v4-ft-ps-e60-moderate. Confirms: seed-2 headline > 0.727 (0.707 + 0.02) at ~1.52x. Fails: <= 0.707, i.e. the seed-1 margin over a0.25 was seed noise.
## Changes
None to code beyond the `parse_arch` fix on this branch (`--arch` accepts any `a<alpha>[_d<depth>]`). Run: `05_distill/main.py --rung C --wsd-max 56000 --wsd-stop 1.3 --seed 2 --runs 'yamnet:a0.50_d8:select:classes=ins_buzz+ambient_rain+human'`.
## Results
WSD curve (seed 2): 7k 0.713, 14k 0.720, 28k 0.725; hit@K +1.08 then +0.61, so the stop rule ended it at 28k (seed 1 ran on to 56k: 0.727 / 0.731 / 0.735 / 0.735). Recorded student: `fe_C_yamnet_a0.50_d8_s2_select_c-buzz-rain-human_wsd28000`, 0.725 @1.51x (x_yamnet200; 1.77x at 20 s).

Against seed 1 (`..._s1_..._wsd56000`, 0.735):

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.829 | 0.820 | -0.009 | 0.009 | 34 |
| 53 | 0.654 | 0.660 | +0.006 | 0.011 | 32 |
| 1_11 | 0.713 | 0.686 | -0.027 | 0.010 | 32 |
| 1_143 | 0.717 | 0.706 | -0.011 | 0.014 | 26 |
| 1_37 | 0.764 | 0.752 | -0.012 | 0.012 | 15 |
- mean: 0.735 → 0.725 (-0.010 ± 0.005); inclusive 0.618 → 0.610
- tiers: loud -0.003, untagged -0.012, background -0.002, quiet +0.002, faint 0

Against a0.25 (`fe_C_yamnet_a0.25_s1_..._wsd56000`, 0.707 @1.62x):

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.819 | 0.820 | +0.001 | 0.015 | 34 |
| 53 | 0.642 | 0.660 | +0.018 | 0.015 | 32 |
| 1_11 | 0.668 | 0.686 | +0.018 | 0.010 | 32 |
| 1_143 | 0.689 | 0.706 | +0.017 | 0.018 | 26 |
| 1_37 | 0.717 | 0.752 | +0.035 | 0.016 | 15 |
- mean: 0.707 → 0.725 (+0.018 ± 0.007); inclusive 0.594 → 0.610
- tiers: loud +0.009, untagged +0.023, background +0.006, quiet +0.024, faint +0.067 (35 frames)

Reading: seed 2 lands 0.010 under seed 1, about twice the eval SD and inside the ~0.01-0.02 seed noise LOOP.md quotes for students. The truncation is not the cause: seed 1 was already 0.735 at 28k. The 1_11 shortfall (-0.027 ± 0.010) eats most of seed 1's +0.057 1_11 gain over a0.25; with seed 2, 1_11 is +0.018 ± 0.010 over a0.25, and that is weak evidence. Against a0.25 every fold moves up or holds. 1_37 moves most (+0.035 ± 0.016, 15 events, so it is unsure).
## Conclusion
Neither outcome, but closer to confirming. Seed 2 at 0.725 is above the fail line (0.707) and just under the confirm line (0.727). Two-seed mean is 0.730 @~1.51x, which is +0.023 over a0.25 (0.707 @1.62x) on a single a0.25 seed. So a0.50_d8 holds as the best student at ~1.5x: it beats the a0.50→a0.25 width line by ~0.02, on every fold, which is more than the ~0.028 the seed-1 run alone suggested. Its 1_11 advantage is about half seed noise. When comparing later d8 widths (a0.375_d8, a0.25_d8, a0.75_d8, all seed 1), use the two-seed mean 0.730 as the a0.50_d8 point, not 0.735.
