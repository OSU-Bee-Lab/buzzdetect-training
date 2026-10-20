# bn-pooled
## Hypothesis
IDEAS #37 (a), pooled AdaBN. The depth8 tail's 14 BN layers keep AudioSet moving statistics while their conv weights fine-tune; the statistics the tail actually sees on this data (both views, plain and octave-up) differ. Re-estimating each BN layer's moving mean/variance, in graph order, on frames from the rotation's own training folds (never the held-out fold; no labels, no weight change) aligns normalisation with the domain the weights were trained on. It ships as static weights. Prediction: >= +0.02 on the headline over the unadapted model, on the same weights, with `untagged`/`loud` up. Falsifier: within 0.02, or a loss, which would say the fine-tuned weights already absorbed the AudioSet statistics (they were trained under them, so swapping them post hoc may misalign the weights instead). Overlaps `ps-bntrain` (BN trained on the pool, +0.022 two-draw mean at 30 epochs) in what it changes, but here the weights were trained under the old statistics, so the two differ in whether the weights co-adapt.
## Changes
`tools_exp/soup.py --bn-pool 20000` (exp/ps-d8-soup's tool): per fold, after loading the weights, draws 20000 frames uniformly from the fold's `folds_train` (asserted to exclude the held-out fold), and re-estimates each tail BN layer's statistics on their 40000 view maps layer by layer. Applied offline to the saved weights of ps-d8-soup's members: `bn-pooled` = member m1 re-estimated, compared against m1 itself (same weights, exactly matched); `bn-pooled-soup` = the uniform soup re-estimated, against `ps-d8-soup`. No training.

## Results
### bn-pooled vs ps-d8-soup-m1 (same weights, unadapted; exactly matched)
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.579 | 0.043 | -0.536 | 0.074 | 32 |
| 53 | 0.561 | 0.073 | -0.488 | 0.099 | 28 |
| 1_11 | 0.601 | 0.005 | -0.596 | 0.102 | 26 |
| 1_143 | 0.541 | 0.000 | -0.541 | 0.069 | 22 |
| 1_150 | 0.361 | 0.000 | -0.361 | 0.066 | 21 |
| 1_95 | 0.233 | 0.000 | -0.233 | 0.038 | 46 |
| 1_37 | 0.564 | 0.014 | -0.550 | 0.085 | 14 |
| 1_114 | 0.405 | 0.014 | -0.391 | 0.055 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.481 → this 0.019 (-0.462 ± 0.027)
- inclusive (sensitivity), same thresholds: 0.399 → 0.020 (-0.379)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.802 | 0.010 | -0.792 | 122 |
| untagged | 0.497 | 0.021 | -0.476 | 2418 |
| background | 0.483 | 0.056 | -0.427 | 1874 |
| quiet | 0.120 | 0.019 | -0.101 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

### bn-pooled-soup vs ps-d8-soup (same soup, unadapted)
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.436 | 0.041 | -0.395 | 0.090 | 32 |
| 53 | 0.375 | 0.166 | -0.209 | 0.134 | 28 |
| 1_11 | 0.514 | 0.262 | -0.252 | 0.057 | 26 |
| 1_143 | 0.524 | 0.027 | -0.497 | 0.074 | 22 |
| 1_150 | 0.213 | 0.000 | -0.213 | 0.039 | 21 |
| 1_95 | 0.188 | 0.007 | -0.181 | 0.035 | 46 |
| 1_37 | 0.548 | 0.007 | -0.541 | 0.099 | 14 |
| 1_114 | 0.485 | 0.006 | -0.479 | 0.046 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.410 → this 0.064 (-0.346 ± 0.028)
- inclusive (sensitivity), same thresholds: 0.335 → 0.051 (-0.284)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.795 | 0.212 | -0.583 | 122 |
| untagged | 0.451 | 0.062 | -0.389 | 2418 |
| background | 0.260 | 0.096 | -0.164 | 1874 |
| quiet | 0.088 | 0.019 | -0.069 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Buzz logit SD per fold, bn-pooled: 1_29 1.94, 53 0.08, 1_11 0.54, 1_143 1.42, 1_150 3.76, 1_95 3.92, 1_37 4.36, 1_114 0.88 (m1 on 53: 1.04). The scores did not collapse to a constant: on several folds they spread wider than m1's. They just no longer rank buzz above the negatives.

Checks against a tool defect: (1) ps-d8-soup's parity run showed soup.py's build/load/score path reproduces m1 exactly (fold 53, 0 flips, max |logit diff| 6e-6), so the only change here is the BN re-estimation. (2) `reestimate_bn` asserts the layer-by-layer chain equals `tail()` on the pooled maps. (3) `pool_maps` reshapes each stored frame to (2, 6, 4, 512), the same layout as the head's `Reshape((n_ctx,) + _TRUNK_SHAPE)` before `TimeDistributed(tail)`, giving 40000 view maps from 20000 frames out of 48 training folds; the held-out fold is asserted out. (4) Per-layer mean shifts are small (0.000-0.052, largest at the pointwise BNs), so the maps are real trunk activations, not scrambled ones. Variance changes were not logged.

## Conclusion
Falsified, emphatically. Pooled AdaBN on member m1's own weights takes the headline from 0.481 to 0.019 (-0.462 ± 0.027); every fold drops to <= 0.073 (1_11 -0.596 ± 0.102, 1_143 -0.541 ± 0.069, 1_29 -0.536 ± 0.074, 1_150 -0.361 ± 0.066, 1_95 -0.233 ± 0.038), and every tier falls (loud -0.792, untagged -0.476, background -0.427, quiet -0.101). On the soup it is the same, 0.410 to 0.064 (-0.346 ± 0.028). The falsifier's reading holds: the fine-tuned tail weights were trained under AudioSet's moving statistics (BN frozen, `trainable=False`, gamma/beta included) and co-adapted to them, so swapping in this data's statistics post hoc misaligns every layer's normalisation with the weights after it, compounding over 14 BN layers, even though the mean shifts are small. This does not contradict `ps-bntrain` (+0.022 two-draw mean), where the statistics moved during training and the weights followed; re-estimation only makes sense for weights trained under the statistics being installed. IDEAS #37 (b) and (c) estimate statistics the same post-hoc way, from the held-out site's audio, so they inherit this misalignment on these weights; they stay Luke's call, and this result is evidence against running them on frozen-BN fine-tunes (they would need a model trained with BN statistics tracking, as in `ps-bntrain`).
