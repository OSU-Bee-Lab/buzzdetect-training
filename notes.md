# d1-ens-teacher
## Hypothesis
Distillation arm, IDEAS D1 (Luke approved it, including its `moderate` runs). On `moderate` the logit mean of
three draws of the teacher's recipe scores 0.602 against the deployed teacher's 0.574 (`ens3-moderate`,
+0.028 ± 0.006), because the draws' negatives decorrelate and every fold's threshold drops. A student's speed
does not depend on its teacher's cost, so if a student can learn the averaged logits the gain is free at the
lite tier.

Prediction: `twofast32:a0.50_d8` (rung C, WSD to 56000, buzz+rain+human, seed 1) distilled from the 3-draw
ensemble lands more than 0.02 above the same student of the single teacher (seed 1 0.701, seed 2 0.694, at
5.33x), with background and untagged carrying it as they do in the teacher.
Falsifier: it lands within seed noise (0.02) of 0.701 / 0.694. Then the student's error against one teacher
is already larger than what the ensemble adds, and D1 is closed for this width.

Matched control: `fe_C_twofast32_a0.50_d8_s1_c-buzz-rain-human_wsd56000` (and its seed 2), same rung, schedule,
budget, front end, width and classes. The ensemble is a new teacher, so its rows carry a different `key`
and the comparison is by headline (labels, not teacher agreement); hit@K and lost% are against a
different teacher and are not comparable. One anchor only: the root disk has 108 GB free and a teacher's
packed rung-C shards are 34-46 GB per front end.

## Changes
- `chain_ship.sh`: shipped `-r2` and `-r3` (the `ens3-moderate` draws, whose shipped models were untrained) on
  `moderate`, exported to buzzdetect (`.local/members/`).
- `tools_exp/build_ens_teacher.py`: one ONNX graph that runs the three members' trunks on one waveform and
  outputs the mean of their raw logits minus the ensemble's own activation centers (from the rotations'
  averaged predictions). The code-regression target is member 1's code, the deployed teacher's own, so only
  the logit targets differ from the single-teacher student. Teacher `v4-ft-ps-e60-moderate-ens3`;
  `plan.csv` and `blacklist.txt` copied from the single teacher's manifest, so slices and V pool are identical.
- `chain.sh`: `main.py --teacher v4-ft-ps-e60-moderate-ens3 --rung C --seed 1 --wsd-max 56000 --wsd-stop 1.3`
  on `twofast32:a0.50_d8` (buzz+rain+human, prefix `ens3_`), then the `mmt` re-export. `exp/yam64h16`'s
  `05_distill` code as a patch.

## Results
WSD curve (V pool against the ensemble teacher): headline 0.679 / 0.689 / 0.697 / 0.698 at 7k / 14k / 28k /
56k; hit@K gains +1.52, +1.16, +0.76, so the stop rule reached 56000, the judged budget. The single teacher's
seed-1 curve was 0.663 / 0.673 / 0.687 / 0.701 (mmt rows). Speed 5.33x (mmt, 200 s), as the control.

Against seed 1 of the same student of the single teacher (`fe_C_twofast32_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt`):

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.807 | 0.820 | +0.013 | 0.013 | 34 |
| 53 | 0.674 | 0.656 | -0.018 | 0.013 | 32 |
| 1_11 | 0.628 | 0.612 | -0.016 | 0.015 | 32 |
| 1_143 | 0.673 | 0.664 | -0.009 | 0.015 | 26 |
| 1_37 | 0.725 | 0.737 | +0.012 | 0.020 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.701 → this 0.698 (-0.003 ± 0.007)
- inclusive (sensitivity), same thresholds: 0.590 → 0.590 (+0.000)

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.810 | 0.814 | +0.004 | 603 |
| untagged | 0.705 | 0.699 | -0.006 | 7485 |
| background | 0.701 | 0.704 | +0.003 | 9412 |
| quiet | 0.233 | 0.241 | +0.008 | 2231 |

Against seed 2 (`..._s2_..._mmt`, 0.694): +0.004 ± 0.007; inclusive +0.009; 1_29 +0.018 ± 0.010,
1_11 +0.016 ± 0.014, the rest within their SD; background +0.014, quiet +0.017, loud -0.008, untagged +0.002.

Reading: the ensemble student sits between the two single-teacher seeds (0.694 / 0.701), -0.003 and +0.004
against them, each ± 0.007 eval SD on top of ~0.01-0.02 seed noise. No fold moves beyond its own SD by a
margin worth reading; every fold delta is unsure. The teacher's own gain (+0.028, carried by background and
untagged, thresholds lower on every fold) does not reach the student: background moves +0.003 / +0.014,
untagged -0.006 / +0.002. Early budgets read a little higher (7k 0.679 vs 0.663, 14k 0.689 vs 0.673), so the
ensemble's targets may be slightly easier to learn early, but by 56k the curves meet. One seed.

## Conclusion
Null: falsified as stated. `twofast32:a0.50_d8` distilled from the 3-draw ensemble teacher is 0.698 at 5.33x
against the single teacher's 0.701 / 0.694 (seeds 1 / 2; -0.003 / +0.004 ± 0.007), all tiers within 0.02.
At this width the student's error against one teacher is larger than what averaging three teachers adds;
the ensemble's lower thresholds do not transfer. Not worth its cost (two shipped trainings, a target re-cache,
3x teacher inference) for this student. Untested: a wider student (yamnet / yam64h16 a0.50_d8), whose
smaller distillation error might leave room for the ensemble's gain; the 7k/14k budgets were +0.016 higher,
a hint for short-budget students only.
