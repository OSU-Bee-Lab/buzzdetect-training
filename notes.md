# ps-d8-ensemble
## Hypothesis
Training arm, no new training. Three plain draws of the same recipe (`ps-depth8` 0.515, `ps-depth8-repeat`
0.509, `ps-depth8-r3` 0.497) differ only by training stochasticity, yet single folds scatter up to ~0.1
between them (`ps-depth8-redraw`). Each fold's threshold is set by ~30 negative frames, so part of that
scatter is which negatives a given draw happens to score high. Averaging the draws' `ins_buzz` logits per
held-out frame should pull those idiosyncratic negatives down while buzz frames, which the draws agree on,
keep their score.

Prediction: the 3-draw logit mean scores above the best single draw (0.515), i.e. more than +0.02 over the
members' mean (0.507), with the gain on the folds whose thresholds are set by few odd negatives (`1_95`,
`1_114`, `1_37`). Falsified if it lands at the members' mean (the draws rank frames alike, and the scatter
is in the positives).

Why it matters if true: an ensemble is 3x the inference cost as a standard-tier model, but as a
distillation teacher its cost is paid once, in the target cache; the student's speed is unchanged.

Matched controls: the three member draws themselves (same frames, same folds, same recipe).
Dose: 2-draw means (three pairs) and the 3-draw mean. Secondary, not the hypothesis: a 5-member mean that
adds the two other clean fine-tune depths with predictions on disk (`ps-depth4`, `ps-depth10-repeat`).

## Changes
- `tools_exp/ensemble_predictions.py`: reads each member's per-fold `predictions.csv`, refuses unless the rows
  match (correct, loudness, sample), writes the mean `ins_buzz` logit as a model directory and scores it with
  `03_train/sx.py`'s `summarize_folds`, as a trained run is scored. No training, no extraction, nothing in
  `03_train/` changed.
- Members (all `yamnet_trunk_pitchshift` on `medium`, 30 epochs, rows identical across the five):
  `ps-depth8` (0.515), `ps-depth8-repeat` (0.509), `ps-depth8-r3` (0.497); secondary `ps-depth4` (0.523),
  `ps-depth10-repeat` (0.487).
- Models written: `ps-d8-ens3` (the hypothesis), `ps-d8-ens2ab/ac/bc` (pairs), `ps-d8-ens5` (all five),
  `ps-ens2-d4d10` (the two non-depth8 runs only: members disjoint from `ens3`, so an independent replication).

## Results
Headlines (`sensitivity_exclquiet` at fpr 0.005, mean over the 8 rotating folds):

| model | members | headline | members' mean | gain |
|---|---|---|---|---|
| single draws | d8 / repeat / r3 | 0.515 / 0.509 / 0.497 | 0.507 | |
| `ps-d8-ens2ab` | d8 + repeat | 0.556 | 0.512 | +0.044 |
| `ps-d8-ens2ac` | d8 + r3 | 0.544 | 0.506 | +0.038 |
| `ps-d8-ens2bc` | repeat + r3 | 0.538 | 0.503 | +0.035 |
| **`ps-d8-ens3`** | d8 + repeat + r3 | **0.558** | 0.507 | **+0.051** |
| `ps-ens2-d4d10` | depth4 + depth10-repeat | 0.559 | 0.505 | +0.054 |
| `ps-d8-ens5` | all five | 0.575 | 0.506 | +0.069 |

`tools/results.py ps-depth8-repeat ps-d8-ens3` (the median draw as the matched control):

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.568 | 0.619 | +0.051 | 0.024 | 32 |
| 53 | 0.522 | 0.581 | +0.059 | 0.041 | 28 |
| 1_11 | 0.585 | 0.632 | +0.047 | 0.027 | 26 |
| 1_143 | 0.626 | 0.640 | +0.014 | 0.029 | 22 |
| 1_150 | 0.380 | 0.470 | +0.090 | 0.041 | 21 |
| 1_95 | 0.236 | 0.312 | +0.076 | 0.022 | 46 |
| 1_37 | 0.641 | 0.701 | +0.060 | 0.027 | 14 |
| 1_114 | 0.512 | 0.509 | -0.003 | 0.029 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.509 → this 0.558 (+0.049 ± 0.011)
- inclusive (sensitivity), same thresholds: 0.421 → 0.467 (+0.046)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.819 | 0.861 | +0.042 | 122 |
| untagged | 0.531 | 0.580 | +0.049 | 2418 |
| background | 0.444 | 0.507 | +0.063 | 1874 |
| quiet | 0.126 | 0.162 | +0.036 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Against the other two members: `ps-depth8` +0.043 ± 0.010 (every fold up: `1_29` +0.039 ± 0.024, `53` +0.021
± 0.020, `1_11` +0.067 ± 0.024, `1_143` +0.036 ± 0.032, `1_150` +0.072 ± 0.039, `1_95` +0.041 ± 0.021, `1_37`
+0.036 ± 0.023, `1_114` +0.033 ± 0.033; loud +0.066, untagged +0.043, background +0.032, quiet +0.015);
`ps-depth8-r3` +0.061 ± 0.011 (every fold up: `1_29` +0.133 ± 0.031, `53` +0.031 ± 0.022, `1_11` +0.059 ±
0.020, `1_143` +0.054 ± 0.043, `1_150` +0.035 ± 0.041, `1_95` +0.028 ± 0.017, `1_37` +0.053 ± 0.031, `1_114`
+0.095 ± 0.040; loud +0.005, untagged +0.056, background +0.101, quiet +0.032).

`ps-d8-ens5` against `ps-d8-ens3`: +0.017 ± 0.007 (0.575; incl. quiet 0.479 vs 0.467); `1_114` +0.067 ± 0.027,
`1_143` +0.036 ± 0.026, `53` +0.019 ± 0.014, the rest within ± 0.006; loud +0.015, untagged +0.016, background
+0.007.

Per fold, members and ensembles:

| fold | d8 | repeat | r3 | d4 | d10-rep | ens3 | ens5 | ens2-d4d10 |
|---|---|---|---|---|---|---|---|---|
| 1_29 | 0.580 | 0.568 | 0.486 | 0.508 | 0.560 | 0.619 | 0.613 | 0.584 |
| 53 | 0.560 | 0.522 | 0.550 | 0.646 | 0.542 | 0.581 | 0.600 | 0.627 |
| 1_11 | 0.565 | 0.585 | 0.573 | 0.606 | 0.617 | 0.632 | 0.648 | 0.642 |
| 1_143 | 0.604 | 0.626 | 0.586 | 0.649 | 0.578 | 0.640 | 0.676 | 0.658 |
| 1_150 | 0.398 | 0.380 | 0.435 | 0.278 | 0.275 | 0.470 | 0.472 | 0.361 |
| 1_95 | 0.271 | 0.236 | 0.284 | 0.309 | 0.240 | 0.312 | 0.312 | 0.323 |
| 1_37 | 0.665 | 0.641 | 0.648 | 0.609 | 0.656 | 0.701 | 0.702 | 0.702 |
| 1_114 | 0.476 | 0.512 | 0.414 | 0.581 | 0.425 | 0.509 | 0.576 | 0.576 |

Mechanism check (three depth8 draws, all held-out frames): the fold thresholds fall on every fold, mean
-1.45 / -1.44 / -1.47 for the members to -1.68 for `ens3` (-1.73 for `ens5`); `1_95` -0.94 / -0.92 / -0.77 to
-1.07. Between draws, a negative frame's logit correlates 0.62 and a buzz frame's 0.88 (between-draw SD 0.35
on negatives against an across-frame SD of 0.56; 0.42 against 1.29 on buzz).

Interpretation. The headline is +0.049 ± 0.011 over the median member and +0.043 ± 0.010 over the best one,
4 SDs of eval sampling; training stochasticity on a headline delta (~0.007) does not change that reading,
and the members' own spread is 0.018. The prediction (above the best single draw, more than +0.02 over the
members' mean) is met at +0.051. It replicates on members that share nothing with `ens3`: depth4 +
depth10-repeat score 0.559 from 0.523 and 0.487. All three depth8 pairs gain +0.035 to +0.044, the third
member adds ~+0.01 to the pairs' 0.546 mean, and two more members of other depths add +0.017 ± 0.007: most
of the gain arrives with the second member, and it has not flattened at five.

Folds: `ens3` beats the best of its three members on seven of eight folds, so this is more than averaging
out a bad draw. Against the median member, `1_95` +0.076 ± 0.022, `1_150` +0.090 ± 0.041, `1_37` +0.060 ±
0.027, `1_29` +0.051 ± 0.024 and `1_11` +0.047 ± 0.027 are about 2 SDs or more; `53` +0.059 ± 0.041 and
`1_143` +0.014 ± 0.029 are within theirs (unsure); `1_114` is flat (-0.003 ± 0.029) against that member but
+0.033 and +0.095 against the other two, so unsure there. The fold prediction was only partly right: the
gain is not confined to `1_95` / `1_114` / `1_37`; it is spread over every fold, hard and rich alike.
`1_150` and `1_95`, the two lowest folds, each end above every single run on disk except `ps-depth4`'s
`1_95` (0.309 vs 0.312, a tie).

Tiers against the median member: loud +0.042 (122 frames), untagged +0.049, background +0.063, quiet +0.036.
Untagged is up against all three members (+0.043 to +0.056) and background against all three (+0.032 to
+0.101); loud is up against two and flat against the third (+0.005). Isolated buzz and the drones both gain.

The mechanism is the predicted one: negatives are where the draws disagree (correlation 0.62 against 0.88
on buzz), so the mean lowers every fold's threshold by ~0.2 logits while buzz frames keep their place.

Not selection on the scored metric: members were chosen before scoring (the three same-recipe draws; the
five are every clean fine-tune run with predictions on disk), nothing was tuned, and no epoch or weight
was picked on the held-out folds.

## Conclusion
Positive, the largest clean step since ps-depth8: the logit mean of three ps-depth8 draws is 0.558 (0.467
incl. quiet) against the members' 0.497-0.515, +0.049 ± 0.011 over the median draw, replicated by a
disjoint depth4 + depth10 pair (0.559 from 0.523 / 0.487); five members reach 0.575. Every fold's
threshold drops (negatives decorrelate between draws, buzz does not), `1_95` +0.076 ± 0.022 and `1_150`
+0.090 ± 0.041 against the median draw, untagged +0.049, background +0.063, loud +0.042. It is a training
budget lever, not a new representation: N draws cost N trainings. For the standard tier the members share
YAMNet layers 1-7 (frozen in depth8), so an ensemble need not cost N full passes (untimed). For the lite
tier a student's speed does not depend on its teacher's size, so an ensemble teacher is the cheap place to
spend this; that is a new teacher (train N on `moderate`, deploy one averaged graph, re-cache targets),
queued in IDEAS.md. Also untested: whether one run can get the same decorrelation (EMA within a run did
not: `ps-ema` -0.008), and where the curve flattens past five members.
