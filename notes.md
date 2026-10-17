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
