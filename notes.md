# mixstyle-xfold
## Hypothesis
IDEAS #31, the cross-deployment form. `mixstyle-dep` drew MixStyle partners from the frame's own deployment (-0.031 +- 0.014 vs ps-depth8-r3), which only mixes within-site style, not what differs between deployments. With the augmentation rule now drawn at the rotation (03_train/CLAUDE.md, 2026-10-19), partners can come from anywhere in the rotation's training pool (k-1 training folds, never the held-out fold). Same setting otherwise (p=0.5, Beta(0.1,0.1), per-channel mean/std over the cached 6x4 layer-7 map) on the ps-depth8 recipe, so the comparison against mixstyle-dep isolates where partners come from. Prediction: >= +0.02 over the three-draw depth8 mean (0.507), with 1_95 / 1_114 / 1_150 and `untagged`/`loud` up. Falsifier: within ~0.02 of 0.507, flat or falling 1_95 / 1_114, or the same `loud`/`untagged` loss mixstyle-dep showed. If it lands at or below mixstyle-dep, the MixStyle line on this map closes (a gentler p would only interpolate toward the control).
## Changes
`TRUNK_MIXSTYLE_XFOLD=1` in `03_train/train.py` (on top of mixstyle-dep's `TRUNK_MIXSTYLE`): partner index drawn uniformly over the rotation's whole training buffer instead of within the frame's fold. The buffer only ever holds the rotation's training folds. Run: `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 TRUNK_MIXSTYLE=0.5 TRUNK_MIXSTYLE_XFOLD=1`, embedder `yamnet_trunk_pitchshift_depth8`, 30 epochs, medium.
## Results
### vs ps-depth8-r3 (matched control)
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.456 | -0.030 | 0.039 | 32 |
| 53 | 0.550 | 0.585 | +0.035 | 0.025 | 28 |
| 1_11 | 0.573 | 0.560 | -0.013 | 0.034 | 26 |
| 1_143 | 0.586 | 0.577 | -0.009 | 0.047 | 22 |
| 1_150 | 0.435 | 0.343 | -0.092 | 0.039 | 21 |
| 1_95 | 0.284 | 0.239 | -0.045 | 0.025 | 46 |
| 1_37 | 0.648 | 0.602 | -0.046 | 0.055 | 14 |
| 1_114 | 0.414 | 0.122 | -0.292 | 0.066 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.436 (-0.061 ± 0.015)
- inclusive (sensitivity), same thresholds: 0.411 → 0.355 (-0.056)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.826 | -0.030 | 122 |
| untagged | 0.524 | 0.454 | -0.070 | 2418 |
| background | 0.406 | 0.423 | +0.017 | 1874 |
| quiet | 0.130 | 0.116 | -0.014 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

### vs mixstyle-dep (within-fold partners)
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.484 | 0.456 | -0.028 | 0.038 | 32 |
| 53 | 0.592 | 0.585 | -0.007 | 0.030 | 28 |
| 1_11 | 0.528 | 0.560 | +0.032 | 0.038 | 26 |
| 1_143 | 0.559 | 0.577 | +0.018 | 0.029 | 22 |
| 1_150 | 0.361 | 0.343 | -0.018 | 0.037 | 21 |
| 1_95 | 0.239 | 0.239 | +0.000 | 0.021 | 46 |
| 1_37 | 0.585 | 0.602 | +0.017 | 0.039 | 14 |
| 1_114 | 0.377 | 0.122 | -0.255 | 0.055 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.466 → this 0.436 (-0.030 ± 0.013)
- inclusive (sensitivity), same thresholds: 0.387 → 0.355 (-0.032)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.804 | 0.826 | +0.022 | 122 |
| untagged | 0.484 | 0.454 | -0.030 | 2418 |
| background | 0.444 | 0.423 | -0.021 | 1874 |
| quiet | 0.138 | 0.116 | -0.022 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Buzz logit SD per fold 0.51-1.16 (1_150 lowest at 0.51), no collapse.

Headline 0.436 vs ps-depth8-r3 0.497: -0.061 +- 0.015, about four SDs down, and -0.071 against the depth8 three-draw mean (0.507). Against mixstyle-dep (0.466) it is -0.030 +- 0.013, so drawing partners from other deployments cost more than drawing them within the deployment, not less.

Per fold vs ps-depth8-r3: 1_114 -0.292 +- 0.066 is the dominant loss (0.414 -> 0.122), and it is also -0.255 +- 0.055 against mixstyle-dep, so cross-deployment mixing is specifically what hurts it. 1_150 -0.092 +- 0.039 (hard fold, down). 1_95 -0.045 +- 0.025 and 1_37 -0.046 +- 0.055 down but within about two SDs, 1_37 unsure. 1_29 -0.030 +- 0.039, 1_11, 1_143 within SD (unsure). 53 +0.035 +- 0.025 is the only fold up, weak. Against mixstyle-dep, folds other than 1_114 are all within their SDs (unsure).

Tiers vs control: untagged -0.070 (2418 frames), loud -0.030 (122), quiet -0.014, background +0.017. The falsifier's untagged loss is repeated and larger; background is again the only tier up, as in mixstyle-dep.

## Conclusion
Falsified. Cross-deployment MixStyle on the layer-7 map is worse than the within-fold version (-0.030 +- 0.013) and well below control (-0.061 +- 0.015), driven by 1_114 collapsing (-0.292 +- 0.066) and untagged -0.070; 1_150 and 1_95 also down. Swapping deployment style statistics at layer 7 appears to destroy signal the head needs (channel statistics there carry content, not only recorder style). The MixStyle line on this map closes: a gentler p would only interpolate toward the control.
