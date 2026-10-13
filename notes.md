# ens3-moderate
## Hypothesis
Training arm, the confirmation step of IDEAS D1 (an ensemble teacher; Luke approved it, including its
`moderate` runs, 2026-10-11). On `medium`, the logit mean of three draws of one fine-tuned-trunk recipe is
+0.049 ± 0.011 over the median draw (`ps-d8-ensemble`), because negatives decorrelate between draws (0.62)
and buzz does not (0.88), so every fold's threshold drops. `moderate` has ~5x the data and the teacher's
recipe is depth12 at 60 epochs, so draws may agree more and the gain may shrink.

Prediction: the logit mean of three draws of the teacher's recipe on `moderate` (`v4-ft-ps-e60-moderate`,
0.574, plus two new draws) scores more than +0.02 over the members' mean, with thresholds lower on every fold.
Falsifier: it lands within 0.02 of the members' mean (on `moderate` the draws rank negatives alike); then D1's
teacher is not worth N trainings and a target re-cache.
Matched controls: the three member draws (same frames, folds, recipe). Dose: the three 2-draw means.
Also reported: the spread of the three draws on `moderate` (five `medium` draws spanned 0.464-0.515).

## Changes
- No pipeline change. `chain.sh` trains `v4-ft-ps-e60-moderate-r2` and `-r3` with the first draw's command
  (`tools/queue_v4_moderate.sh`: `yamnet_trunk_pitchshift_depth12`, 60 epochs, `TRUNK_LR_BACKBONE=1e-5
  TRUNK_LR_HEAD=2e-4 TRUNK_BATCH=1024 TRUNK_FP16=1 TRUNK_STREAM=1`), rotations only, no shipped model.
- `tools_exp/ensemble_predictions.py` (from `exp/ps-d8-ensemble`, unchanged) writes `ens3-moderate` and the
  three pairs. Draw 1 is main's `models/v4-ft-ps-e60-moderate` (trained 2026-09-27; `03_train` commits since
  then add opt-in env switches and a fold-line print, no change to this recipe's training).
