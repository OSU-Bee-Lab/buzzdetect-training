# ps2sf-d8-r2
## Hypothesis
Repeat draw of `ps2-seamfree-d8` (0.509; +0.012 ± 0.017 vs ps-depth8-r3), whose conclusion asked for one before anyone builds on it. Its one large move was the trill fold 1_114 at 0.670 (+0.256 ± 0.069), above every earlier depth8-family run (max 0.512), offset by weak losses at 1_150 (-0.092 ± 0.057) and 1_95 (-0.037 ± 0.031).

Prediction: 1_114 holds above 0.55 (i.e. well above the 0.41-0.51 range of the two-view ps-depth8 draws), and the headline lands within ~0.02 of 0.509. Falsifier: 1_114 back inside the ps-depth8 range, which would make the first draw's 1_114 a lucky draw. The 1_150 loss is read the same way: real if it repeats.

Control: `ps-depth8-r3` (matched; the comparator ps2-seamfree-d8 used), with ps2-seamfree-d8 as the first draw.

## Changes
None: same embedder (`yamnet_trunk_pitchshift2sf_depth8`, cache already extracted) and settings (`TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=512 TRUNK_ACCUM=2 SCORE_CHUNK=512`, 30 epochs). Chain: `exp/pssf-d8:chain_b43.sh`.

## Results
vs ps-depth8-r3 (matched control). Buzz logit SD per fold 0.62-1.33 (no collapse).

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.526 | +0.040 | 0.043 | 32 |
| 53 | 0.550 | 0.599 | +0.049 | 0.026 | 28 |
| 1_11 | 0.573 | 0.567 | -0.006 | 0.049 | 26 |
| 1_143 | 0.586 | 0.667 | +0.081 | 0.053 | 22 |
| 1_150 | 0.435 | 0.324 | -0.111 | 0.051 | 21 |
| 1_95 | 0.284 | 0.226 | -0.058 | 0.038 | 46 |
| 1_37 | 0.648 | 0.620 | -0.028 | 0.055 | 14 |
| 1_114 | 0.414 | 0.465 | +0.051 | 0.040 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.499 (+0.002 ± 0.016)
- inclusive (sensitivity), same thresholds: 0.411 → 0.411 (+0.000)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.900 | +0.044 | 122 |
| untagged | 0.524 | 0.518 | -0.006 | 2418 |
| background | 0.406 | 0.464 | +0.058 | 1874 |
| quiet | 0.130 | 0.116 | -0.014 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Repeat vs the first draw (`tools/results.py ps2-seamfree-d8 ps2sf-d8-r2`): headline 0.509 → 0.499 (-0.010 ± 0.015); 1_114 0.670 → 0.465 (-0.205 ± 0.066); every other fold within ±0.066 of draw 1 (1_29 +0.058 ± 0.039, 53 +0.066 ± 0.035, rest inside their SD).

Interpretation: headline +0.002 ± 0.016 vs the control, i.e. no effect. 1_114 at 0.465 (+0.051 ± 0.040 vs control, unsure) is back inside the ps-depth8 two-view range (0.41-0.51): the pre-registered falsifier. The first draw's 0.670 was a lucky draw, a 3-SD swing between two identical runs, which is itself a warning about how wide 1_114's training spread is on this trunk. The hard-fold losses did repeat: 1_150 -0.111 ± 0.051 (draw 1: -0.092 ± 0.057) and 1_95 -0.058 ± 0.038 (draw 1: -0.037 ± 0.031). Gains at 53 (+0.049 ± 0.026) and 1_143 (+0.081 ± 0.053) are weak. Tiers: background +0.058, loud +0.044 (122 frames), untagged -0.006, quiet -0.014.

Pooled over the two draws: headline ~0.504 (+0.007 vs control, within noise); 1_114 ~0.57, which still leans above control but by less than half the first draw's claim; 1_150 ~-0.10 and 1_95 ~-0.05 in both draws.

## Conclusion
Null on the headline (+0.002 ± 0.016); the first draw's 1_114 jump (0.670) did not repeat (0.465, falsifier met). What does repeat is a loss at the two hardest folds (1_150 ≈ -0.10, 1_95 ≈ -0.05 in both draws). The seam-free two-octave view is not worth building on as-is; pssf-d8 and ps12sf-d8 (same batch) show the same hard-fold loss pattern.
