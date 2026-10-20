# pauc-l03
## Hypothesis
IDEAS #33. The headline is one point (FPR 0.005) on the ROC curve, and BCE spends its gradient on all negatives equally. A one-way partial-AUC auxiliary term (Zhu et al. 2022, SOPA-style) ranks each `ins_buzz` positive above the currently highest-scoring 5% of negatives in its batch (squared hinge, margin 1), naming no confuser class. Added at λ=0.3 to the weighted BCE on the ps-depth8 recipe, it should lift the headline >= 0.02 over the three-draw ps-depth8 mean (0.507; matched control ps-depth8-r3 0.497), with the gain on the folds whose thresholds are set by confusers (1_95 aircraft, 1_114 trill, 1_37 background). Falsifier: within ~0.02 or below; a collapsed `buzz logit SD` (~0.1) is an artifact, not a verdict. λ=0.3 is the upper rung of the idea's 0.1/0.3 ladder, run first because a null at the high dose closes the family more cheaply than a null at the low one.
## Changes
- `03_train/train_utils.py`: `pauc_aux_loss(base, buzz_index, lam, top_frac=0.05, margin=1.0)`. Per batch: the squared hinge between each `ins_buzz` positive's logit (raw target > 0.5, `_quiet` included) and each of the top ceil(0.05·n_neg) negative logits on the buzz neuron, averaged over pairs. It is added to every sample's weighted BCE, so the batch mean carries it once. Static shapes (sort and masks), so it traces under XLA.
- `03_train/train.py`: `_loss()` wraps the compiled loss; `TRAIN_PAUC_LAMBDA=<lam>` turns it on (unset = byte-identical to main).
- Run: ps-depth8 recipe (`yamnet_trunk_pitchshift_depth8`, TRUNK_LR_BACKBONE=1e-5, TRUNK_FP16=1, TRUNK_BATCH=1024, 30 epochs) with TRAIN_PAUC_LAMBDA=0.3, from batch 38's chain (`chain_b38.sh`).

## Results
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.379 | -0.107 | 0.044 | 32 |
| 53 | 0.550 | 0.476 | -0.074 | 0.042 | 28 |
| 1_11 | 0.573 | 0.534 | -0.039 | 0.035 | 26 |
| 1_143 | 0.586 | 0.586 | +0.000 | 0.050 | 22 |
| 1_150 | 0.435 | 0.296 | -0.139 | 0.048 | 21 |
| 1_95 | 0.284 | 0.163 | -0.121 | 0.030 | 46 |
| 1_37 | 0.648 | 0.568 | -0.080 | 0.041 | 14 |
| 1_114 | 0.414 | 0.244 | -0.170 | 0.040 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.406 (-0.091 ± 0.015)
- inclusive (sensitivity), same thresholds: 0.411 → 0.330 (-0.081)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.817 | -0.039 | 122 |
| untagged | 0.524 | 0.433 | -0.091 | 2418 |
| background | 0.406 | 0.306 | -0.100 | 1874 |
| quiet | 0.130 | 0.120 | -0.010 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Buzz logit SD per fold: 1.28, 1.14, 0.88, 0.91, 0.81, 0.93, 0.93, 1.54. No collapse; the class output is live, at or above the trained-head range.
val_loss (which includes the pAUC term) bottoms at epoch 1-5 on every fold and climbs to 30; not comparable with the control's BCE-only val_loss.

Headline: 0.406 vs matched control ps-depth8-r3 0.497, **-0.091 ± 0.015** (eval sampling; ~±0.02 with training noise). Against the three-draw depth8 mean 0.507 it is -0.101. Far outside noise: a real loss.

Per fold (each against its own eval SD): 1_114 -0.170 ± 0.040, 1_150 -0.139 ± 0.048, 1_95 -0.121 ± 0.030, 1_29 -0.107 ± 0.044, 1_37 -0.080 ± 0.041, 53 -0.074 ± 0.042: all well beyond their SDs. 1_11 -0.039 ± 0.035 (unsure, leaning down), 1_143 +0.000 ± 0.050 (flat). The hard and confuser-set folds this was aimed at (1_95, 1_114, 1_37, 1_150) lost the most, not the least.

Tiers: background -0.100, untagged -0.091, loud -0.039, quiet -0.010 (flat, 627 frames). The loss is in the moderate-loudness bulk, not the quiet tail.

## Conclusion
Falsified at λ=0.3: -0.091 ± 0.015 vs ps-depth8-r3, -0.101 vs the depth8 three-draw mean, with the largest losses on the target folds (1_114, 1_150, 1_95). Not a collapse (logit SD 0.8-1.5). Pushing positives above the batch's top-5% negatives by a margin of 1 appears to stretch the buzz logit scale and pull the threshold-setting negatives up with it, rather than separating them; the same direction as the closed `pairwise-rank` and `tail-loss` results. IDEAS #33 is trimmed to its untested λ=0.1 rung; a decline there too would close the low-FPR-loss family (the idea's own falsifier).
