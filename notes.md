# ps-depth8-redraw
## Hypothesis
`pseudo-label` (2026-10-10) left an open thread: its teacher, meant to be a third draw of the `ps-depth8` recipe, scored 0.466 where the two earlier draws scored 0.515 (2026-10-01) and 0.509. If nothing in `03_train` or the medium set drifted since 2026-10-01, a plain `ps-depth8` CV from current main lands with the earlier draws (within ~2 SD of a headline delta, ~0.03, of 0.512) and the 0.466 belongs to the `PSEUDO_SET` code path or to an unlucky draw. Falsified if this run lands near 0.466: then main's recipe has drifted, the era's best control is stale, and the cause needs finding before anything else is built on it.

Either way this run is the fresh matched control for the batch's next training experiments (controls go stale; `docs/training-proposer-facts.md`).

Control: `ps-depth8-repeat` (0.509), with `ps-depth8` (0.515) and `pseudo-label_teacher` (0.466) as the other draws.

## Changes
None. Main at e43b4c6, existing `yamnet_trunk_pitchshift_depth8` cache.
Run: `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, `--embedder yamnet_trunk_pitchshift_depth8 --epochs 30`, set medium, translation general.

## Results
All 8 folds trained (`buzz logit SD` 0.57-1.21; `1_150` at 0.57 is the lowest and is the same in `ps-d8-taps`, not a collapse). Against `ps-depth8-repeat`:

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.568 | 0.486 | -0.082 | 0.040 | 32 |
| 53 | 0.522 | 0.550 | +0.028 | 0.045 | 28 |
| 1_11 | 0.585 | 0.573 | -0.012 | 0.028 | 26 |
| 1_143 | 0.626 | 0.586 | -0.040 | 0.046 | 22 |
| 1_150 | 0.380 | 0.435 | +0.055 | 0.046 | 21 |
| 1_95 | 0.236 | 0.284 | +0.048 | 0.028 | 46 |
| 1_37 | 0.641 | 0.648 | +0.007 | 0.034 | 14 |
| 1_114 | 0.512 | 0.414 | -0.098 | 0.036 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.509 → this 0.497 (-0.012 ± 0.014)
- inclusive (sensitivity), same thresholds: 0.421 → 0.411 (-0.010)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.819 | 0.856 | +0.037 | 122 |
| untagged | 0.531 | 0.524 | -0.007 | 2418 |
| background | 0.444 | 0.406 | -0.038 | 1874 |
| quiet | 0.126 | 0.130 | +0.004 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Other draws (same tool): against `ps-depth8` 0.515 → 0.497 (-0.018 ± 0.012); against `pseudo-label_teacher` 0.466 → 0.497 (+0.031 ± 0.015).

Interpretation. The headline is within one SD of `ps-depth8-repeat` and 1.5 eval SDs below `ps-depth8`; with training stochasticity added (~0.014 on a headline delta) neither is a difference. The three plain draws are 0.515, 0.509, 0.497: mean 0.507, spread 0.009, in line with the measured run-to-run SD. So main still reproduces the recipe; nothing drifted to ~0.47.

The `pseudo-label` teacher's 0.466 sits 0.031 ± 0.015 below this draw and ~0.04 below the three-draw mean: about 2 SD of a single delta, more against the mean. That is weak-to-moderate evidence that the `PSEUDO_SET` code path's stage-1 teacher is not the same recipe as plain `ps-depth8` (or it is a ~2 SD unlucky draw; one sample cannot separate them). Its deficit against this run is in `1_150` (+0.111 ± 0.050), `1_95` (+0.083 ± 0.041) and `1_37` (+0.077 ± 0.047), tier `untagged` (+0.026) and `loud`, not `background`.

Per fold against `ps-depth8-repeat`, the moves past their eval SD are `1_114` -0.098 ± 0.036, `1_29` -0.082 ± 0.040 (down) and `1_95` +0.048 ± 0.028, `1_150` +0.055 ± 0.046 (up). These are two draws of one recipe, so this is what fold-level run-to-run scatter looks like on the fine-tuned trunk: up to ~0.1 on a fold, well above the 0.016-0.026 measured on the frozen `--hidden 1024` head. Per-fold deltas of that size between single runs of this recipe are not evidence on their own. Tiers: `background` -0.038 follows `1_29`; `loud` +0.037 rests on 122 frames; `untagged` flat.

## Conclusion
Main reproduces `ps-depth8`: 0.497 against 0.509 and 0.515 (-0.012 ± 0.014 vs `ps-depth8-repeat`). The era's best control is not stale. `pseudo-label`'s 0.466 teacher is ~2 SD low and may belong to the `PSEUDO_SET` path; its student-vs-teacher null stands either way since both shared that path. Fold-level scatter between identical fine-tuned runs reaches ~0.1 (`1_114`, `1_29`), so single-run fold deltas on this recipe need a repeat before they are read. `ps-depth8-r3` is the matched control for `ps-d8-taps` and `ps-d8-foldbal`.
