# ps-depth10-repeat
## Hypothesis
ps-depth10 (fine-tune layers 10-14) gave +0.021 +/- 0.013 over v4-ft-ps: under the MDE and one draw. The loop's rule is to confirm a gain with one repeat run before building on it. Prediction: if real, the repeat lands near +0.02 with the same signature (1_37 up, 1_114 down); if it is training noise it regresses to ~0.

Control: `v4-ft-ps` (0.452) and `ps-depth10` (0.473, first draw), identical config (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
None. Same code and cache (`yamnet_trunk_pitchshift_depth10`); only the training draw differs.

## Results
Repeat draw, identical config. vs v4-ft-ps:

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.499 | 0.560 | +0.061 | 0.025 | 32 |
| 53 | 0.517 | 0.542 | +0.025 | 0.030 | 28 |
| 1_11 | 0.575 | 0.617 | +0.042 | 0.032 | 26 |
| 1_143 | 0.559 | 0.578 | +0.019 | 0.038 | 22 |
| 1_150 | 0.312 | 0.275 | -0.037 | 0.037 | 21 |
| 1_95 | 0.201 | 0.240 | +0.039 | 0.022 | 46 |
| 1_37 | 0.480 | 0.656 | +0.176 | 0.051 | 14 |
| 1_114 | 0.471 | 0.425 | -0.046 | 0.028 | 28 |

Headline 0.452 -> 0.487 (+0.035 +/- 0.012); inclusive 0.372 -> 0.402. vs first draw ps-depth10 (0.473): +0.014 +/- 0.011. Two-draw mean 0.480 (+0.028 over control). Tiers vs control: background +0.050, untagged +0.034, quiet +0.017, loud -0.017.

Both draws: 1_37 up (+0.104, +0.176), 1_29 up (+0.058, +0.061), 1_114 down (-0.065, -0.046), 1_95 slightly up (+0.028, +0.039), 1_150 flat/down (+0.003, -0.037). Tier gain sits in background and untagged; loud is flat-to-down.

## Conclusion
The depth-10 gain reproduces: two draws, +0.021 and +0.035 (mean +0.028), each ~1.7-2.9 SD. A modest, real headline lift via the background and untagged tiers; 1_37 and 1_29 carry it, 1_114 pays. Hard folds 1_150 and 1_114 are not helped (1_95 weakly up). Depth is a real but small lever; the open question is whether going deeper (depth 8) extends it.
