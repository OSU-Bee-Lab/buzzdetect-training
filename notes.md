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

## Results
All ten new rotations trained (buzz logit SD 0.71-1.15). `ensemble_predictions.py` accepted draw 1's
predictions (rows match). `sensitivity_exclquiet` at fpr 0.005, 5 rotating folds of `moderate`:

| model | headline | incl. quiet |
|---|---|---|
| draw 1 `v4-ft-ps-e60-moderate` | 0.574 | 0.470 |
| draw 2 `-r2` | 0.556 | 0.455 |
| draw 3 `-r3` | 0.564 | 0.460 |
| members' mean | 0.565 | 0.462 |
| `ens2ab` / `ens2ac` / `ens2bc` | 0.594 / 0.595 / 0.591 | 0.488 / 0.488 / 0.485 |
| `ens3-moderate` | **0.602** | 0.494 |

`ens3` against the members' mean: **+0.037** (against each draw: +0.028 ± 0.006, +0.046 ± 0.005,
+0.038 ± 0.006, eval-sampling SD only). The pairs are +0.026 to +0.031 over their own two members, so the third
draw adds about +0.008 on top of two.

Draw spread: 0.556-0.574 (range 0.018; r2 - draw 1 is -0.018 ± 0.006, r3 - draw 1 is -0.010 ± 0.007), tighter
than `medium`'s five draws (0.464-0.515). The teacher (draw 1) is the best of the three.

Per fold, `ens3` against the three draws (delta ± eval SD against draw 1 / r2 / r3):

| fold | draws | members' mean | ens3 | vs draw 1 | vs r2 | vs r3 |
|---|---|---|---|---|---|---|
| 1_29 | 0.562 / 0.539 / 0.542 | 0.548 | 0.606 | +0.044 ± 0.015 | +0.067 ± 0.012 | +0.064 ± 0.013 |
| 53 | 0.512 / 0.530 / 0.506 | 0.516 | 0.547 | +0.035 ± 0.012 | +0.017 ± 0.012 | +0.041 ± 0.022 |
| 1_11 | 0.612 / 0.607 / 0.614 | 0.611 | 0.635 | +0.023 ± 0.011 | +0.028 ± 0.010 | +0.021 ± 0.010 |
| 1_143 | 0.645 / 0.603 / 0.616 | 0.621 | 0.640 | -0.005 ± 0.010 | +0.037 ± 0.011 | +0.024 ± 0.012 |
| 1_37 | 0.539 / 0.501 / 0.543 | 0.528 | 0.583 | +0.044 ± 0.013 | +0.082 ± 0.014 | +0.040 ± 0.011 |

Every fold is above its members' mean; one cell is not a gain (1_143 against draw 1, -0.005 ± 0.010, unsure:
draw 1 is the high draw there). No fold falls.

Thresholds (logit at fpr 0.005), draws against `ens3`: lower than all three members on every fold.

| fold | draw 1 | r2 | r3 | ens3 |
|---|---|---|---|---|
| 1_29 | -1.783 | -1.672 | -1.657 | -1.886 |
| 53 | -1.165 | -1.312 | -1.324 | -1.373 |
| 1_11 | -1.849 | -1.942 | -1.807 | -2.029 |
| 1_143 | -1.267 | -1.184 | -1.195 | -1.323 |
| 1_37 | -1.558 | -1.448 | -1.481 | -1.704 |

Tiers, members' mean -> `ens3`: background 0.441 -> 0.495 (+0.054), untagged 0.595 -> 0.630 (+0.035),
loud 0.798 -> 0.811 (+0.013), quiet 0.172 -> 0.186 (+0.014), faint 0.067 flat. The gain is largest on
background (1_29, 53) and untagged; loud is nearly flat against r2/r3 and +0.029 against draw 1.

`tools/results.py` output, `ens3-moderate` against each draw, then r2 and r3 against draw 1:

```
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.562 | 0.606 | +0.044 | 0.015 | 34 |
| 53 | 0.512 | 0.547 | +0.035 | 0.012 | 32 |
| 1_11 | 0.612 | 0.635 | +0.023 | 0.011 | 32 |
| 1_143 | 0.645 | 0.640 | -0.005 | 0.010 | 26 |
| 1_37 | 0.539 | 0.583 | +0.044 | 0.013 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.574 → this 0.602 (+0.028 ± 0.006)
- inclusive (sensitivity), same thresholds: 0.470 → 0.494 (+0.024)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.782 | 0.811 | +0.029 | 603 |
| untagged | 0.607 | 0.630 | +0.023 | 7485 |
| background | 0.444 | 0.495 | +0.051 | 9412 |
| quiet | 0.177 | 0.186 | +0.009 | 2231 |
| faint | 0.067 | 0.067 | +0.000 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.539 | 0.606 | +0.067 | 0.012 | 34 |
| 53 | 0.530 | 0.547 | +0.017 | 0.012 | 32 |
| 1_11 | 0.607 | 0.635 | +0.028 | 0.010 | 32 |
| 1_143 | 0.603 | 0.640 | +0.037 | 0.011 | 26 |
| 1_37 | 0.501 | 0.583 | +0.082 | 0.014 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.556 → this 0.602 (+0.046 ± 0.005)
- inclusive (sensitivity), same thresholds: 0.455 → 0.494 (+0.039)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.805 | 0.811 | +0.006 | 603 |
| untagged | 0.582 | 0.630 | +0.048 | 7485 |
| background | 0.448 | 0.495 | +0.047 | 9412 |
| quiet | 0.168 | 0.186 | +0.018 | 2231 |
| faint | 0.067 | 0.067 | +0.000 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.542 | 0.606 | +0.064 | 0.013 | 34 |
| 53 | 0.506 | 0.547 | +0.041 | 0.022 | 32 |
| 1_11 | 0.614 | 0.635 | +0.021 | 0.010 | 32 |
| 1_143 | 0.616 | 0.640 | +0.024 | 0.012 | 26 |
| 1_37 | 0.543 | 0.583 | +0.040 | 0.011 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.564 → this 0.602 (+0.038 ± 0.006)
- inclusive (sensitivity), same thresholds: 0.460 → 0.494 (+0.034)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.807 | 0.811 | +0.004 | 603 |
| untagged | 0.597 | 0.630 | +0.033 | 7485 |
| background | 0.432 | 0.495 | +0.063 | 9412 |
| quiet | 0.170 | 0.186 | +0.016 | 2231 |
| faint | 0.067 | 0.067 | +0.000 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.562 | 0.539 | -0.023 | 0.013 | 34 |
| 53 | 0.512 | 0.530 | +0.018 | 0.018 | 32 |
| 1_11 | 0.612 | 0.607 | -0.005 | 0.010 | 32 |
| 1_143 | 0.645 | 0.603 | -0.042 | 0.014 | 26 |
| 1_37 | 0.539 | 0.501 | -0.038 | 0.014 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.574 → this 0.556 (-0.018 ± 0.006)
- inclusive (sensitivity), same thresholds: 0.470 → 0.455 (-0.015)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.782 | 0.805 | +0.023 | 603 |
| untagged | 0.607 | 0.582 | -0.025 | 7485 |
| background | 0.444 | 0.448 | +0.004 | 9412 |
| quiet | 0.177 | 0.168 | -0.009 | 2231 |
| faint | 0.067 | 0.067 | +0.000 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.562 | 0.542 | -0.020 | 0.015 | 34 |
| 53 | 0.512 | 0.506 | -0.006 | 0.023 | 32 |
| 1_11 | 0.612 | 0.614 | +0.002 | 0.010 | 32 |
| 1_143 | 0.645 | 0.616 | -0.029 | 0.015 | 26 |
| 1_37 | 0.539 | 0.543 | +0.004 | 0.014 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.574 → this 0.564 (-0.010 ± 0.007)
- inclusive (sensitivity), same thresholds: 0.470 → 0.460 (-0.010)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.782 | 0.807 | +0.025 | 603 |
| untagged | 0.607 | 0.597 | -0.010 | 7485 |
| background | 0.444 | 0.432 | -0.012 | 9412 |
| quiet | 0.177 | 0.170 | -0.007 | 2231 |
| faint | 0.067 | 0.067 | +0.000 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.
```

## Conclusion
Prediction held: the 3-draw logit mean on `moderate` is 0.602, +0.037 over its members' mean (0.565) and
+0.028 ± 0.006 over the best member (the deployed teacher), with a lower threshold than every member on every
fold. The gain is smaller than `medium`'s (+0.049 ± 0.011 over the median draw; here +0.038 ± 0.006 over the
median draw), as expected with ~5x the data, but well clear of the 0.02 falsifier. Two draws already give
+0.026 to +0.031; the third adds ~+0.008. Tiers: background and untagged carry it, loud and quiet up slightly,
nothing down. Draws on `moderate` span only 0.018. The training-arm confirmation of D1 is done; whether the
gain survives distillation into a student is still untested.
