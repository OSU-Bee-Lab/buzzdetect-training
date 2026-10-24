# sf-ens-d8
## Hypothesis
Same-recipe ps-depth8 ensembles saturate at three (`ps-d8-ens3` 0.558 = `ps-d8-ens-same5`), and a cross-recipe member only helped when weak members were not swapped in: `plain-d8` (0.418) cost -0.024 to -0.032 against ens3, its 1_114 collapse dragging the mean. `plain-d8`'s conclusion: "ensemble diversity has to come from members that are individually as strong." `ps2-seamfree-d8` is such a member (0.509, at the depth8 three-draw mean) with a different fold profile (1_114 0.670 vs ps draws' 0.41-0.51; 1_150 weaker).

Prediction: swapping one ps-depth8 draw for a seam-free member raises the triple above ens3 by >= +0.02, with 1_114 up and 1_150 not down beyond its SD. Falsifier: the cross-recipe triples at or below ens3, as plain-d8's were.

Disclosure: one triple (ps2-seamfree-d8 + ps-depth8 + ps-depth8-repeat) was scored as a row-compatibility probe before this section was written (0.570, i.e. +0.012 over ens3). The prediction stands as written and every triple below is reported, not only that one.

Design (eval only, `tools_exp/ensemble_predictions.py` from `exp/ps-d8-ensemble`, logit mean): the three swap-one triples {sf + two of a/b/c} against ens3 and against the ps pair alone; once this batch's runs finish, the same with `ps2sf-d8-r2`, both seam-free draws together with one ps draw, and the 4-member {a, b, sf, sf-r2} against ens3. a = ps-depth8, b = ps-depth8-repeat, c = ps-depth8-r3.

## Changes
None to training. `run_ens.sh` builds every pre-registered combination with `tools_exp/ensemble_predictions.py` (ins_buzz logit mean; refuses unless members scored identical rows) and scores each with `tools/results.py ps-d8-ens3 <ens>`. Raw outputs in `ens_results/`. a = ps-depth8, b = ps-depth8-repeat, c = ps-depth8-r3, sf = ps2-seamfree-d8 (0.509), r2 = ps2sf-d8-r2 (0.499).

## Results
Headline vs ps-d8-ens3 (a+b+c, 0.558), ± eval-sampling SD of the paired delta:

| ensemble | headline | Δ vs ens3 | Δ vs its ps pair |
|---|---|---|---|
| a+b | 0.556 | -0.002 ± 0.007 | |
| a+c | 0.544 | -0.014 ± 0.008 | |
| b+c | 0.538 | -0.020 ± 0.008 | |
| sf+a+b (probe, disclosed) | 0.570 | +0.012 ± 0.010 | +0.014 |
| sf+a+c | 0.579 | +0.021 ± 0.010 | +0.035 |
| sf+b+c | 0.564 | +0.006 ± 0.010 | +0.026 |
| r2+a+b | 0.573 | +0.015 ± 0.009 | +0.017 |
| r2+a+c | 0.566 | +0.008 ± 0.008 | +0.022 |
| r2+b+c | 0.566 | +0.008 ± 0.009 | +0.028 |
| sf+r2+a | 0.577 | +0.019 ± 0.013 | |
| sf+r2+b | 0.563 | +0.005 ± 0.012 | |
| sf+r2+c | 0.571 | +0.013 ± 0.012 | |
| a+b+sf+r2 (4) | 0.587 | +0.029 ± 0.011 | |
| sf+r2 | 0.542 | -0.016 ± 0.016 | |

Per-fold deltas vs ens3 (ens3 per fold: 1_29 0.619, 53 0.581, 1_11 0.632, 1_143 0.640, 1_150 0.470, 1_95 0.312, 1_37 0.701, 1_114 0.509):

| ensemble | 1_29 | 53 | 1_11 | 1_143 | 1_150 | 1_95 | 1_37 | 1_114 |
|---|---|---|---|---|---|---|---|---|
| sf+a+b | +0.000 | -0.013 | -0.009 | +0.081 | -0.063 | -0.020 | -0.003 | +0.123 |
| sf+a+c | -0.018 | +0.017 | -0.004 | +0.081 | -0.021 | +0.000 | +0.005 | +0.106 |
| sf+b+c | -0.039 | +0.007 | +0.017 | +0.028 | -0.030 | -0.028 | -0.009 | +0.098 |
| r2+a+b | +0.029 | +0.023 | +0.016 | +0.046 | -0.044 | +0.014 | -0.008 | +0.045 |
| r2+a+c | +0.013 | +0.038 | +0.000 | +0.037 | -0.044 | +0.004 | +0.017 | -0.002 |
| r2+b+c | +0.000 | +0.009 | +0.031 | +0.045 | -0.016 | +0.007 | -0.004 | -0.007 |
| sf+r2+a | +0.005 | +0.022 | +0.002 | +0.045 | -0.056 | -0.003 | +0.001 | +0.135 |
| sf+r2+b | -0.001 | +0.009 | -0.005 | +0.046 | -0.063 | -0.035 | -0.004 | +0.090 |
| sf+r2+c | -0.019 | +0.023 | +0.012 | +0.055 | -0.035 | -0.021 | +0.005 | +0.085 |
| a+b+sf+r2 | +0.012 | +0.016 | +0.016 | +0.073 | -0.046 | +0.021 | +0.004 | +0.133 |
| sf+r2 | -0.044 | +0.034 | -0.041 | +0.036 | -0.102 | -0.052 | -0.068 | +0.113 |

Logged entry (a+b+sf+r2, `models/sfe-sfr2-ab`), full results.py:

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.619 | 0.631 | +0.012 | 0.020 | 32 |
| 53 | 0.581 | 0.597 | +0.016 | 0.024 | 28 |
| 1_11 | 0.632 | 0.648 | +0.016 | 0.030 | 26 |
| 1_143 | 0.640 | 0.713 | +0.073 | 0.030 | 22 |
| 1_150 | 0.470 | 0.424 | -0.046 | 0.032 | 21 |
| 1_95 | 0.312 | 0.333 | +0.021 | 0.033 | 46 |
| 1_37 | 0.701 | 0.705 | +0.004 | 0.034 | 14 |
| 1_114 | 0.509 | 0.642 | +0.133 | 0.037 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.558 → this 0.587 (+0.029 ± 0.011)
- inclusive (sensitivity), same thresholds: 0.467 → 0.492 (+0.025)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.861 | 0.884 | +0.023 | 122 |
| untagged | 0.580 | 0.608 | +0.028 | 2418 |
| background | 0.507 | 0.522 | +0.015 | 1874 |
| quiet | 0.162 | 0.166 | +0.004 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Interpretation. Swapping one ps draw for one seam-free member: the six swap-one triples average +0.012 vs ens3 (sf +0.013, r2 +0.010; range +0.006 to +0.021, each about 1 SD). Against the ps pair they replace, adding a seam-free third member gives +0.014 to +0.035 (mean +0.024), while ens-same5 = ens3 says a third/fourth ps draw adds ~0. So a seam-free member is a diverse, useful ensemble member even though alone it is no better than a ps draw, which is what `plain-d8`'s conclusion predicted for a member as strong as the others. That's weak-to-moderate evidence: no single triple clears 2 SD, but all six land on the same side.

Prediction (>= +0.02) met by one of six triples (sf+a+c); the falsifier (cross-recipe at or below ens3) is not met. The 1_114 gain is mostly sf's: sf triples +0.098 to +0.123 at 1_114, r2 triples -0.007 to +0.045, which follows the members' own 1_114 draws (0.670 vs 0.465). It is a member's luck carrying through, not a reliable recipe effect. Every seam-free combination loses at 1_150 (-0.016 to -0.063 for triples, -0.046 ± 0.032 for the 4-member), inherited from both seam-free draws' -0.10 there, so the "1_150 not down beyond its SD" condition holds for the 4-member and most triples only marginally. 1_143 rises in every combination (+0.028 to +0.081).

The 4-member a+b+sf+r2 is the best figure (0.587, +0.029 ± 0.011), with tiers untagged +0.028, loud +0.023, background +0.015 and quiet flat. It is the top of 14 scored configurations (selection inflates it); size alone doesn't explain it (5 same-recipe members = 3). Pre-registered, but the honest per-swap estimate is the +0.012 family mean.

## Conclusion
Cross-recipe ensembling with seam-free members helps a little: one seam-free member in place of a ps draw gives +0.012 on average over six triples (each ~1 SD), +0.024 over the ps pair it joins; the 4-member a+b+sf+r2 reaches 0.587 (+0.029 ± 0.011, best of 14, selection-inflated). Gains at 1_143 and (sf only) 1_114; 1_150 loses in every combination (members' -0.10 there). Ensemble diversity from a pitch-view recipe change works where plain-d8's didn't, because the members are equally strong.
