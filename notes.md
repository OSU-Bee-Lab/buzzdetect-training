# pitch-aux-d8
## Hypothesis
`binary-d8` showed the 14 non-buzz classes' supervision shapes trunk features the buzz boundary uses (collapsing them cost -0.116 at e30). Buzz is the one class still collapsed. Adding pitch subclasses (`buzz_low`/`buzz_medium`/`buzz_high`) as extra multi-label outputs beside the scored `ins_buzz` (unchanged targets, unchanged loss weight) gives the fine-tuned tail more structure within buzz and should sharpen the buzz/non-buzz boundary.

Prediction (IDEAS #38): headline >= +0.02 over the matched control `ps-depth8-r3` (0.497) with `untagged` or a hard fold (1_95/1_114/1_150) up. Falsifier: within ~0.02.

Control: `ps-depth8-r3` (same ps-depth8 cache, recipe, 30 epochs, batch 1024; ran 2026-10-21 era-cohort with plain-d8/sam-d8; medium annotations unchanged since).

## Changes
- Translation `general_pitch` (`02_set/sets/medium/translate.R`, `translations/general_pitch.csv`): pitch-tagged buzz labels map to `ins_buzz|buzz_low`, `ins_buzz|buzz_medium` or `ins_buzz|buzz_high`; untagged-pitch buzz maps to `ins_buzz` alone. Every other row is `general`'s.
- `train_utils.build_classes` and `dataset.labels_to_targets` split a `to` value on `|`, so one label can set several multi-hot targets.
- `train._pin_primary_weights`: every original class keeps the loss weight it gets under the translation with the auxiliary targets dropped (`[pin]` lines in `chain_b42.log`; `ins_buzz` 0.27 in both). The auxiliary outputs take `build_weights`' inverse-frequency weight: `buzz_high` 1.6, `buzz_low` 0.9, `buzz_medium` 0.5. Together they put ~3x `ins_buzz`'s weight on within-buzz pitch supervision. A smaller auxiliary weight is untested.
- Same ps-depth8 cache, recipe, 30 epochs and batch 1024 as the control `ps-depth8-r3`. The shipped model was not trained.

## Results
Buzz logit SD per fold: 1.15, 1.00, 0.66, 0.74, 0.55, 0.77, 0.83, 0.84. 1_150 (0.55) is just below the 0.65-1.2 band but nowhere near collapse.

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.473 | -0.013 | 0.034 | 32 |
| 53 | 0.550 | 0.560 | +0.010 | 0.039 | 28 |
| 1_11 | 0.573 | 0.611 | +0.038 | 0.022 | 26 |
| 1_143 | 0.586 | 0.587 | +0.001 | 0.050 | 22 |
| 1_150 | 0.435 | 0.347 | -0.088 | 0.045 | 21 |
| 1_95 | 0.284 | 0.233 | -0.051 | 0.028 | 46 |
| 1_37 | 0.648 | 0.697 | +0.049 | 0.035 | 14 |
| 1_114 | 0.414 | 0.438 | +0.024 | 0.026 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.493 (-0.004 ± 0.013)
- inclusive (sensitivity), same thresholds: 0.411 → 0.408 (-0.003)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.820 | -0.036 | 122 |
| untagged | 0.524 | 0.524 | +0.000 | 2418 |
| background | 0.406 | 0.400 | -0.006 | 1874 |
| quiet | 0.130 | 0.145 | +0.015 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only; with training stochasticity, the headline delta's full SD is about 0.014.

The headline delta is -0.004 ± 0.013, well within noise, so this is a null result. The falsifier ("within ~0.02") holds.

Per fold, the two hardest moved down. 1_150 fell -0.088 ± 0.045 (about 2 SD, still one draw, and its buzz logit SD was the lowest at 0.55). 1_95 fell -0.051 ± 0.028 (about 1.8 SD). The gains were 1_37 at +0.049 ± 0.035 (unsure), 1_11 at +0.038 ± 0.022 and 1_114 at +0.024 ± 0.026 (unsure). The rest are within their SD. If anything, the hard-fold pattern runs against the hypothesis: that is weak evidence that pitch supervision hurts the folds it was meant to help. Tiers: untagged is flat (+0.000), loud fell -0.036 on only 122 frames (unsure), and quiet rose +0.015 (unsure).

## Conclusion
Adding pitch subclasses as auxiliary outputs beside `ins_buzz` does not move the headline (-0.004 ± 0.013). untagged is flat. The hard folds 1_150 (-0.088 ± 0.045) and 1_95 (-0.051 ± 0.028) went down, not up. `binary-d8`'s lesson, that extra non-buzz classes shape useful trunk features, does not carry over to splitting buzz itself, at least at the inverse-frequency auxiliary weight used here (~3x `ins_buzz`'s). Not worth a repeat. The cheap frozen-probe arm from IDEAS #38, which tests a different mechanism (max over three buzz hyperplanes), was not run.
