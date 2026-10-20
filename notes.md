# ps-d8-soup
## Hypothesis
IDEAS #34. The 3-draw logit ensemble is the log's largest clean step (`ps-d8-ensemble`, +0.049 over the median draw), and its source is decorrelated negatives between draws. Model soups (Wortsman et al. 2022) average fine-tuned weights that share a pretrained init and often match the ensemble at one model's cost; the depth8 tail starts from the same YAMNet layers 8-14 in every draw, but each draw's head (Dense) starts from its own random init, so the soup may leave the shared basin. Prediction: the uniform soup of three depth8 draws scores at or above its best member and closes most of the gap to the logit ensemble of the same three. Falsifier: soup below its best member (the draws left the shared basin; a soup gives one graph's cost only if it is at least as good as one draw). WiSE-FT on the soup (tail moved a fraction A in {0.25, 0.5, 0.75} back toward pretrained YAMNet, head kept): a gain at small A would say the 30-epoch tail over-adapts; monotone loss with A says the adaptation is all signal, as `llrd-depth4` suggested.
## Changes
`TRUNK_SAVE_WEIGHTS=1` in `03_train/train.py`: each rotation also writes its final weights (`folds/<fold>/fold.weights.h5`). `tools_exp/soup.py`: per fold, the elementwise mean of the members' weights (optionally WiSE-interpolated toward the pretrained tail), scored on the held-out fold through `train._score_fold`, summarised by `sx.py`. `tools_exp/ensemble_predictions.py` (from exp/ps-d8-ensemble): logit-mean ensemble of the same members. Members: three ps-depth8-recipe CVs (`TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, `yamnet_trunk_pitchshift_depth8`, 30 epochs, medium) named ps-d8-soup-m1/m2/m3. The members are also three more draws of the depth8 recipe.

## Results
### soup vs best member (ps-d8-soup-m1)
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.579 | 0.436 | -0.143 | 0.041 | 32 |
| 53 | 0.561 | 0.375 | -0.186 | 0.096 | 28 |
| 1_11 | 0.601 | 0.514 | -0.087 | 0.038 | 26 |
| 1_143 | 0.541 | 0.524 | -0.017 | 0.037 | 22 |
| 1_150 | 0.361 | 0.213 | -0.148 | 0.068 | 21 |
| 1_95 | 0.233 | 0.188 | -0.045 | 0.023 | 46 |
| 1_37 | 0.564 | 0.548 | -0.016 | 0.047 | 14 |
| 1_114 | 0.405 | 0.485 | +0.080 | 0.034 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.481 → this 0.410 (-0.071 ± 0.019)
- inclusive (sensitivity), same thresholds: 0.399 → 0.335 (-0.064)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.802 | 0.795 | -0.007 | 122 |
| untagged | 0.497 | 0.451 | -0.046 | 2418 |
| background | 0.483 | 0.260 | -0.223 | 1874 |
| quiet | 0.120 | 0.088 | -0.032 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

### soup vs logit ensemble of the same three (ps-d8-soup-ens)
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.530 | 0.436 | -0.094 | 0.031 | 32 |
| 53 | 0.611 | 0.375 | -0.236 | 0.114 | 28 |
| 1_11 | 0.649 | 0.514 | -0.135 | 0.043 | 26 |
| 1_143 | 0.613 | 0.524 | -0.089 | 0.047 | 22 |
| 1_150 | 0.424 | 0.213 | -0.211 | 0.085 | 21 |
| 1_95 | 0.250 | 0.188 | -0.062 | 0.029 | 46 |
| 1_37 | 0.647 | 0.548 | -0.099 | 0.048 | 14 |
| 1_114 | 0.526 | 0.485 | -0.041 | 0.043 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.531 → this 0.410 (-0.121 ± 0.022)
- inclusive (sensitivity), same thresholds: 0.444 → 0.335 (-0.109)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.885 | 0.795 | -0.090 | 122 |
| untagged | 0.554 | 0.451 | -0.103 | 2418 |
| background | 0.470 | 0.260 | -0.210 | 1874 |
| quiet | 0.150 | 0.088 | -0.062 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

### WiSE A=0.25 vs soup
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.436 | 0.345 | -0.091 | 0.027 | 32 |
| 53 | 0.375 | 0.269 | -0.106 | 0.029 | 28 |
| 1_11 | 0.514 | 0.456 | -0.058 | 0.022 | 26 |
| 1_143 | 0.524 | 0.541 | +0.017 | 0.035 | 22 |
| 1_150 | 0.213 | 0.139 | -0.074 | 0.037 | 21 |
| 1_95 | 0.188 | 0.160 | -0.028 | 0.017 | 46 |
| 1_37 | 0.548 | 0.455 | -0.093 | 0.031 | 14 |
| 1_114 | 0.485 | 0.495 | +0.010 | 0.024 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.410 → this 0.358 (-0.052 ± 0.010)
- inclusive (sensitivity), same thresholds: 0.335 → 0.288 (-0.047)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.795 | 0.795 | +0.000 | 122 |
| untagged | 0.451 | 0.401 | -0.050 | 2418 |
| background | 0.260 | 0.144 | -0.116 | 1874 |
| quiet | 0.088 | 0.064 | -0.024 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

### ensemble vs ps-depth8-r3
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.530 | +0.044 | 0.027 | 32 |
| 53 | 0.550 | 0.611 | +0.061 | 0.027 | 28 |
| 1_11 | 0.573 | 0.649 | +0.076 | 0.033 | 26 |
| 1_143 | 0.586 | 0.613 | +0.027 | 0.044 | 22 |
| 1_150 | 0.435 | 0.424 | -0.011 | 0.067 | 21 |
| 1_95 | 0.284 | 0.250 | -0.034 | 0.021 | 46 |
| 1_37 | 0.648 | 0.647 | -0.001 | 0.040 | 14 |
| 1_114 | 0.414 | 0.526 | +0.112 | 0.048 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.531 (+0.034 ± 0.014)
- inclusive (sensitivity), same thresholds: 0.411 → 0.444 (+0.033)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.885 | +0.029 | 122 |
| untagged | 0.524 | 0.554 | +0.030 | 2418 |
| background | 0.406 | 0.470 | +0.064 | 1874 |
| quiet | 0.130 | 0.150 | +0.020 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

### Members (each vs ps-depth8-r3, 0.497)
- m1 0.481 (-0.016 ± 0.014), m2 0.470 (-0.027 ± 0.013), m3 0.472 (-0.025 ± 0.013). Member mean 0.474; the six-draw depth8 mean (ps-depth8 0.515, -repeat 0.509, -r3 0.497, m1-m3) is 0.491, draw SD ~0.019. These three sit lower than the first three; same recipe, plus only `TRUNK_SAVE_WEIGHTS=1`, which writes weights after scoring and cannot change them.
- WiSE: w25 0.358 (-0.052 ± 0.010 vs soup), w50 0.252 (-0.158 ± 0.016), w75 0.142 (-0.268 ± 0.021): monotone loss with A.

### Logit SDs and a parity check
Buzz logit SD per fold, soup 0.32-0.63 and w25 0.22-0.42, against ~1.0 for a member (m1 on 53: 1.04, soup on 53: 0.42). The soup's logits shrink by about 2.5x: the averaged weights land between the members, not in a shared basin.
To rule out a tool defect, soup.py rescored m1 alone (no averaging, no WiSE, no AdaBN) on fold 53 (test model, since deleted): 7540 frames, max |logit diff| 6e-6, 0 detections flipped at logit > 0, sens 0.561 = m1's. The load/build/score path is exact, so the losses are the treatment.

## Conclusion
Falsified. The uniform soup scores 0.410, -0.071 ± 0.019 below its best member (m1, 0.481) and -0.121 ± 0.022 below the logit ensemble of the same three members (0.531). Per fold vs m1: 53 -0.186 ± 0.096, 1_150 -0.148 ± 0.068, 1_29 -0.143 ± 0.041, 1_11 -0.087 ± 0.038, 1_95 -0.045 ± 0.023 down; 1_143 and 1_37 within SD; 1_114 +0.080 ± 0.034 up, the only fold where the soup beats m1 (it is also below the ensemble there, -0.041 ± 0.043). The hard folds 1_150 and 1_95 both lose. With logits shrunk ~2.5x, the three draws did not stay in one basin: the random-init heads and 30 epochs at 1e-5 on the tail are enough to take them apart, so averaging weights gives an interpolation between models rather than a cheaper ensemble.
WiSE-FT toward pretrained YAMNet layers loses monotonically with A (-0.052, -0.158, -0.268 at 0.25/0.5/0.75), so the tail's fine-tuning is signal, not over-adaptation; this agrees with `llrd-depth4`.
Side results: the logit ensemble is again the best depth8 number, 0.531 (+0.034 ± 0.014 vs ps-depth8-r3, +0.057 over the median member 0.472), repeating `ps-d8-ensemble`'s step on three new draws. The three new draws (0.481/0.470/0.472) bring the depth8 recipe's six-draw mean to 0.491.
