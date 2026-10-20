# lp-ft
## Hypothesis
IDEAS #30 (LP-FT, Kumar et al. 2022). depth8's tail trains from epoch 1 under a randomly initialised Dense head, so the first gradients into layers 8-14 come from a random readout: the feature-distortion setting LP-FT targets. Every held-out fold is out-of-distribution by construction. Training the head alone for 10 epochs with the tail frozen (linear-probe phase), then the usual 30-epoch fine-tune (tail at 1e-5, head at 2e-4, fresh optimizer), should lift the headline >= 0.02 over the three-draw ps-depth8 mean (0.507; matched control ps-depth8-r3 0.497), with the gain on `untagged` and the hard folds (1_95, 1_114, 1_150), the most OOD deployments. Falsifier: within ~0.02 of 0.507. Caveat known in advance: the LP phase adds 10 head-only epochs on top of the 30, so the total budget differs (e15/e60 show the FT budget matters; the head-only epochs do not move the tail).

## Changes
`03_train/train.py`: `TRUNK_LP_EPOCHS=N` trains the head alone (tail frozen) for N epochs, then rebuilds a fresh optimizer and runs the usual fine-tune (tail 1e-5, head 2e-4). Run: N=10, 30 FT epochs, `yamnet_trunk_pitchshift_depth8`, medium, `TRUNK_FP16=1 TRUNK_BATCH=1024`.

## Results
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.498 | +0.012 | 0.044 | 32 |
| 53 | 0.550 | 0.511 | -0.039 | 0.028 | 28 |
| 1_11 | 0.573 | 0.561 | -0.012 | 0.021 | 26 |
| 1_143 | 0.586 | 0.631 | +0.045 | 0.034 | 22 |
| 1_150 | 0.435 | 0.296 | -0.139 | 0.045 | 21 |
| 1_95 | 0.284 | 0.240 | -0.044 | 0.025 | 46 |
| 1_37 | 0.648 | 0.566 | -0.082 | 0.043 | 14 |
| 1_114 | 0.414 | 0.210 | -0.204 | 0.036 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.439 (-0.058 ± 0.013)
- inclusive (sensitivity), same thresholds: 0.411 → 0.362 (-0.049)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.845 | -0.011 | 122 |
| untagged | 0.524 | 0.462 | -0.062 | 2418 |
| background | 0.406 | 0.402 | -0.004 | 1874 |
| quiet | 0.130 | 0.134 | +0.004 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Buzz logit SD 0.50-1.20 on every fold (1_150 lowest at 0.50, below the 0.65-1.2 band but nowhere near collapse at ~0.1); no artifact.

Headline -0.058 ± 0.013 against the matched control ps-depth8-r3 (eval sampling only; ~0.014 with training noise added); -0.068 against the three-draw depth8 mean 0.507. Well outside noise.
Per fold: 1_114 -0.204 ± 0.036 and 1_150 -0.139 ± 0.045 are large, clear losses on two hard folds; 1_37 -0.082 ± 0.043 a loss; 1_95 -0.044 ± 0.025 weak evidence of a loss; 53 -0.039 ± 0.028 unsure-to-weak loss; 1_143 +0.045 ± 0.034 weak evidence of a gain; 1_29 +0.012 and 1_11 -0.012 unsure.
Tiers: the loss is in `untagged` (-0.062); loud, background and quiet are flat. That is the opposite of the prediction (gain on untagged and the hard folds).
Val-loss argmin moved later on the hardest folds (1_150 epoch 15, 1_114 epoch 13 vs 3-8 elsewhere); the LP phase adds 10 head-only epochs to the budget, so total budget is a confound, but e60 (longer FT) did not lose like this, and the head-only epochs do not move the tail.

## Conclusion
Falsified, and clearly negative: LP-FT at N=10 costs -0.058 ± 0.013 on the headline, concentrated in untagged and in the hard folds 1_114 and 1_150. A warm head seems to let the tail fit the training deployments' structure without the random-head regularisation; whatever the mechanism, head warm-starting is not a lever for depth8. Not worth a repeat at this size; a smaller N (2-3) is the only rung left, and not one I'd queue.
