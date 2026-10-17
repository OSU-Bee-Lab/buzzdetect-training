# d1-ens-yamnet
## Hypothesis
IDEAS D1 remainder. `d1-ens-teacher` found `twofast32:a0.50_d8` from the 3-draw ensemble teacher
(`v4-ft-ps-e60-moderate-ens3`, +0.028 over the single teacher on the CV) at 0.698 vs 0.701 / 0.694 from
the single teacher: null, perhaps because that student's distillation error swamps what the ensemble adds.
`yam64h16:a0.50_d8` has the least distillation error of the fast-ish students (single teacher 0.727, ~3.8x).
Prediction: from the ensemble teacher it lands > 0.02 above 0.727 (> 0.747). Falsifier: within 0.02 of the
single-teacher student (seed 1 0.727), which closes D1 across widths.
Matched control: `fe_C_yam64h16_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt` (same rung, WSD, seed, classes).
## Changes
`chain.sh` only (from d1-ens-teacher): ensemble teacher, `yam64h16:a0.50_d8`, prefix `ens3`.
## Results
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.817 | 0.824 | +0.007 | 0.008 | 34 |
| 53 | 0.653 | 0.673 | +0.020 | 0.012 | 32 |
| 1_11 | 0.688 | 0.695 | +0.007 | 0.012 | 32 |
| 1_143 | 0.698 | 0.718 | +0.020 | 0.015 | 26 |
| 1_37 | 0.767 | 0.754 | -0.013 | 0.016 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.725 → this 0.733 (+0.008 ± 0.006)
- inclusive (sensitivity), same thresholds: 0.611 → 0.619 (+0.008)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.823 | 0.820 | -0.003 | 603 |
| untagged | 0.730 | 0.734 | +0.004 | 7485 |
| background | 0.696 | 0.716 | +0.020 | 9412 |
| quiet | 0.256 | 0.263 | +0.007 | 2231 |
| faint | 0.000 | 0.067 | +0.067 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Compared at 28000 steps: the WSD stop rule (hit@K gain < 1.3 twice) ended the ens3 student there (hit@K 75.77, +0.45 over 14000), so no 56000 student exists. The control at 28000 is the matched step budget (its 56000 is 0.727).

## Conclusion
Ensemble-teacher `yam64h16:a0.50_d8` headline 0.733, +0.008 ± 0.006 over the single-teacher control. Prediction (> 0.747) failed; falsifier (within 0.02 of 0.727) met. The ensemble's +0.028 on the teacher does not pass through to a low-distillation-error student either, so D1 is closed across widths. Folds: all within ±0.02 except 1_37 -0.013 (n=15), nothing readable per fold. Background tier +0.020 is the only tier movement, weak evidence.
