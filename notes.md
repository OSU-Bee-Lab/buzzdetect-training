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
## Results
## Conclusion
