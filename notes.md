# mixstyle-dep
## Hypothesis
IDEAS #31. Every augmentation logged so far is a generic perturbation; none models what differs between deployments (recorder response, site acoustics). MixStyle (Zhou et al. 2021) on the cached layer-7 map — with p=0.5 per training frame, each view's per-channel mean/std over the 6x4 map replaced by a Beta(0.1, 0.1) mix of its own and a partner frame's — perturbs feature "style" while keeping content; identity at inference. Partners are drawn from the frame's own training fold (one deployment), per 03_train/CLAUDE.md's no-cross-fold rule, which limits style diversity to within-deployment variation (snips across a deployment's days/hours). On the ps-depth8 recipe it should lift the headline >= 0.02 over the three-draw mean (0.507; matched control ps-depth8-r3 0.497), on `untagged`/`loud` and on 1_95 / 1_114. Falsifier: within ~0.02, flat `untagged`/`loud` and flat 1_95 / 1_114, or the background-tier loss `ps-cutout`/`ps-noise` showed. The cross-training-fold variant (the method's real point) is a policy call for Luke and is not run.
## Changes
`TRUNK_MIXSTYLE=<p>` in `03_train/train.py` (commit e41c6e6): during training of the trunk tail, each view with probability p has its per-channel mean/std over the cached (6,4,512) layer-7 map replaced by a Beta(0.1, 0.1) mix of its own and a partner frame's statistics; partners drawn from the same training fold. Identity at inference. Run on the ps-depth8 recipe (`TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, embedder `yamnet_trunk_pitchshift_depth8`, 30 epochs, medium) with p=0.5.

## Results
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.484 | -0.002 | 0.042 | 32 |
| 53 | 0.550 | 0.592 | +0.042 | 0.027 | 28 |
| 1_11 | 0.573 | 0.528 | -0.045 | 0.023 | 26 |
| 1_143 | 0.586 | 0.559 | -0.027 | 0.052 | 22 |
| 1_150 | 0.435 | 0.361 | -0.074 | 0.044 | 21 |
| 1_95 | 0.284 | 0.239 | -0.045 | 0.021 | 46 |
| 1_37 | 0.648 | 0.585 | -0.063 | 0.037 | 14 |
| 1_114 | 0.414 | 0.377 | -0.037 | 0.055 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.466 (-0.031 ± 0.014)
- inclusive (sensitivity), same thresholds: 0.411 → 0.387 (-0.024)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.804 | -0.052 | 122 |
| untagged | 0.524 | 0.484 | -0.040 | 2418 |
| background | 0.406 | 0.444 | +0.038 | 1874 |
| quiet | 0.130 | 0.138 | +0.008 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Buzz logit SD per fold: 1.10, 1.07, 0.66, 0.72, 0.59, 0.73, 0.74, 0.80. No collapse, though lower than pauc-l03's run (0.81-1.54); val_loss minima at epochs 2-12, as in the depth8 recipe.

Headline: 0.466 vs matched control ps-depth8-r3 0.497, **-0.031 ± 0.014** (eval sampling; ~±0.02 with training noise). Against the three-draw depth8 mean 0.507 it is -0.041. A loss, modest but outside noise; the predicted >= +0.02 gain is clearly absent.
Per fold: 1_150 -0.074 ± 0.044 (weak evidence of a loss), 1_37 -0.063 ± 0.037 (weak), 1_11 -0.045 ± 0.023 and 1_95 -0.045 ± 0.021 (moderate evidence of a loss), 1_114 -0.037 ± 0.055 (unsure), 1_143 -0.027 ± 0.052 (unsure), 1_29 flat, 53 +0.042 ± 0.027 (weak evidence of a gain). The hard folds 1_95 / 1_114 the hypothesis named moved down, not up.
Tiers: `loud` -0.052 and `untagged` -0.040 down; `background` +0.038 up, the opposite of the background-tier loss cutout/noise showed; `quiet` flat.

## Conclusion
Falsified. Within-fold MixStyle (p=0.5, Beta(0.1,0.1)) on the depth8 layer-7 map costs -0.031 ± 0.014 headline vs the matched control (-0.041 vs the three-draw mean), with losses on the targeted hard folds 1_95 (-0.045 ± 0.021) and 1_150 (-0.074 ± 0.044) and on `loud`/`untagged`; only `background` gained (+0.038). Within-deployment style variation is apparently not the nuisance worth removing: mixing it blurs feature statistics that carry buzz signal. The cross-training-fold variant (IDEAS #31's actual pairing, which 03_train/CLAUDE.md's augmentation rule now permits) is untested; this result does not speak to it beyond suggesting that statistic-mixing at layer 7 is costly.
