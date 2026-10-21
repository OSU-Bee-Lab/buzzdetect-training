# binary-d8
## Hypothesis
On a frozen probe, collapsing the 14 non-buzz classes into one `nonbuzz` target was null once ins_buzz's loss weight was pinned (`binary-control`, E3, -0.006). With a fine-tuned depth8 tail the question is different: the auxiliary classes now shape shared features in seven trainable blocks, not just a readout. Two opposite mechanisms are live. (a) Capacity spent separating trill from plane from rain is capacity not spent on the buzz boundary, so binary lifts the headline. (b) The named confuser classes (`ins_trill` at 1_114, `mech_plane` at 1_95) supervise exactly the features that keep those negatives below threshold, so binary hurts the hard folds.

Prediction (for (a)): headline >= +0.02 over the matched control `ps-depth8-r3` (0.497; depth8 three-draw mean 0.507) with `untagged` up. Falsifier: within 0.02, or 1_114/1_95 down beyond their SDs, which would read as (b).

Control: `ps-depth8-r3` (same recipe and cache, `--translation general`).

## Changes
`03_train/train.py`: `PIN_BUZZ_TRANSLATION=general` sets ins_buzz's loss weight, per rotation, to the value `build_weights` gives the same training samples under `general` (labels only; `_pin_buzz_weight`). Without it ins_buzz's weight moves 0.268 -> 1.339 (1_150's rotation) purely from the class count, as `binary-control` found in E3. Run: ps-depth8-r3's recipe (`yamnet_trunk_pitchshift_depth8`, `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, 30 epochs, medium) with `--translation binary`. Chain: `chain_b41.sh` on this branch.

## Results
Every fold's `buzz logit SD` is in range (0.49 at 1_150 to 1.07 at 1_29; plain-d8 runs 0.44 at 1_150 too, so that fold is just low-variance).

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.394 | -0.092 | 0.035 | 32 |
| 53 | 0.550 | 0.366 | -0.184 | 0.076 | 28 |
| 1_11 | 0.573 | 0.617 | +0.044 | 0.023 | 26 |
| 1_143 | 0.586 | 0.505 | -0.081 | 0.048 | 22 |
| 1_150 | 0.435 | 0.250 | -0.185 | 0.068 | 21 |
| 1_95 | 0.284 | 0.118 | -0.166 | 0.041 | 46 |
| 1_37 | 0.648 | 0.541 | -0.107 | 0.062 | 14 |
| 1_114 | 0.414 | 0.256 | -0.158 | 0.052 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.381 (-0.116 ± 0.019)
- inclusive (sensitivity), same thresholds: 0.411 → 0.309 (-0.102)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.797 | -0.059 | 122 |
| untagged | 0.524 | 0.417 | -0.107 | 2418 |
| background | 0.406 | 0.240 | -0.166 | 1874 |
| quiet | 0.130 | 0.087 | -0.043 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only; with training stochasticity the headline delta's full SD is ~0.02, so -0.116 is ~6 SD out: a large, unambiguous loss, the biggest of any depth8 variant so far (pitch-aug -0.046, snr-aug -0.063).

Per fold: the hard folds fell hardest and well beyond their SDs: 1_150 -0.185 ± 0.068, 1_95 -0.166 ± 0.041, 1_114 -0.158 ± 0.052. 53 -0.184 ± 0.076 and 1_37 -0.107 ± 0.062 also fell beyond their SDs; 1_29 -0.092 ± 0.035 and 1_143 -0.081 ± 0.048 likewise. Only willard (1_11) moved up, +0.044 ± 0.023, about 2 SD: weak evidence of a gain on one hard fold, against losses everywhere else.

Tiers: background lost most (-0.166), untagged -0.107, loud -0.059, quiet -0.043. The prediction for (a) needed untagged up; it fell.

## Conclusion
Falsified, and in the direction of mechanism (b). On a frozen probe binary was null (`binary-control`, -0.006); with seven trainable trunk blocks it costs -0.116 ± 0.019, concentrated on the hard folds (1_150, 1_95, 1_114) and the background tier. Reading: once the trunk is trainable, the 14 non-buzz classes supervise features the buzz boundary itself relies on to push named confusers (trill, plane, rain) down; collapsing them into one `nonbuzz` target removes that supervision and the threshold at 0.5% FPR has to rise. Willard's +0.044 ± 0.023 is the one exception, weak. Keep the full class set for any fine-tuned recipe; if anything, more named negative classes is the direction this points.
