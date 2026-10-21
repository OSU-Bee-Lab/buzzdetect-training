# sam-d8
## Hypothesis
Every held-out fold is an unseen deployment, so the headline is a domain-generalization score. Sharpness-aware minimization (Foret et al. 2021, arXiv 2010.01412) seeks flat minima, and flatness is the best-supported single predictor of domain generalization in the DG literature (Cha et al. 2021, SWAD, arXiv 2102.08604; DomainBed SAM rows). This loop's flatness proxies were weight averaging, which acts after training (`ps-ema` -0.008, `ps-d8-soup` far below its members): SAM changes the trajectory itself, taking each step from the worst point in an L2 ball of radius rho around the weights.

Prediction: SAM (rho 0.05, on the trainable tail and head) on the ps-depth8 recipe lifts the headline >= +0.02 over ps-depth8-r3 (0.497) with `untagged` up and no hard fold (1_95/1_114/1_150) down beyond its SD. Falsifier: within 0.02 or a loss.

Control: `ps-depth8-r3` (same recipe, cache, epochs and batch; SAM doubles the gradient cost per step but not the epoch count).

## Changes
`03_train/train.py`: `TRUNK_SAM_RHO=<rho>` replaces the model's Keras train_step with SAM (`_sam_train_step`): gradient at w, perturb all trainable weights by rho g/|g| (one global norm), gradient there, restore, apply the second gradient with the model's own optimizer (per-variable LRs and clipnorm unchanged). BN stays frozen. Run: ps-depth8-r3's recipe with `TRUNK_SAM_RHO=0.05`. Chain: `exp/binary-d8:chain_b41.sh`.

## Results
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.512 | +0.026 | 0.035 | 32 |
| 53 | 0.550 | 0.610 | +0.060 | 0.025 | 28 |
| 1_11 | 0.573 | 0.554 | -0.019 | 0.024 | 26 |
| 1_143 | 0.586 | 0.586 | +0.000 | 0.043 | 22 |
| 1_150 | 0.435 | 0.380 | -0.055 | 0.044 | 21 |
| 1_95 | 0.284 | 0.247 | -0.037 | 0.021 | 46 |
| 1_37 | 0.648 | 0.588 | -0.060 | 0.044 | 14 |
| 1_114 | 0.414 | 0.458 | +0.044 | 0.049 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.492 (-0.005 ± 0.013)
- inclusive (sensitivity), same thresholds: 0.411 → 0.410 (-0.001)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.804 | -0.052 | 122 |
| untagged | 0.524 | 0.514 | -0.010 | 2418 |
| background | 0.406 | 0.464 | +0.058 | 1874 |
| quiet | 0.130 | 0.130 | +0.000 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Buzz logit SD per fold 0.44-1.07 (1_150 lowest, as in plain-d8/ps-depth8-r3): no collapse. SAM moved val_loss minima later (epochs 7-30 vs the control family's 2-8 on several folds), consistent with a slower, smoothed trajectory.

Headline -0.005 ± 0.013 (± ~0.015 with training noise): no effect. Per fold: 53 +0.060 ± 0.025 (weak evidence up), 1_114 +0.044 ± 0.049 (unsure), 1_29 +0.026 ± 0.035 (unsure); 1_37 -0.060 ± 0.044 and 1_150 -0.055 ± 0.044 (weak evidence down, unsure), 1_95 -0.037 ± 0.021 (weak evidence down — a hard fold), 1_11 -0.019 ± 0.024 (unsure). Two of three hard folds went down, the third (1_114) up within its SD. Tiers: background +0.058, loud -0.052 (122 frames, thin), untagged -0.010.

## Conclusion
Falsified: SAM rho 0.05 on the ps-depth8 recipe is within 0.02 of the control (-0.005 ± 0.013) and costs ~3x the training time (3.8 h CV vs ~1.2 h). Flatness via SAM, like the weight-averaging proxies (ps-ema, ps-d8-soup), does not buy cross-deployment generalization here. Hard folds 1_95 (-0.037 ± 0.021) and 1_150 (-0.055 ± 0.044) leaned down; background tier +0.058 is the only gain worth a glance. One rho only; a smaller rho was not tried and is unlikely to be worth 3x compute.
