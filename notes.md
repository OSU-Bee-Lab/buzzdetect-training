# ps-depth8-repeat
## Hypothesis
ps-depth8's +0.063 over control (+0.028 over depth10-repeat) is a single draw; repeat it to confirm before building further on the depth ladder.

Control: `v4-ft-ps` (0.452) and ps-depth10 / ps-depth8 draws; config identical to ps-depth8 (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
None: repeat of ps-depth8 on the existing yamnet_trunk_pitchshift_depth8 cache.

## Results
Run 2026-10-01 (folds_sx.csv 23:34), left unlogged when the loop switched to the distillation arm; logged 2026-10-06 by batch 24.

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.499 | 0.568 | +0.069 | 0.023 | 32 |
| 53 | 0.517 | 0.522 | +0.005 | 0.040 | 28 |
| 1_11 | 0.575 | 0.585 | +0.010 | 0.029 | 26 |
| 1_143 | 0.559 | 0.626 | +0.067 | 0.045 | 22 |
| 1_150 | 0.312 | 0.380 | +0.068 | 0.051 | 21 |
| 1_95 | 0.201 | 0.236 | +0.035 | 0.023 | 46 |
| 1_37 | 0.480 | 0.641 | +0.161 | 0.062 | 14 |
| 1_114 | 0.471 | 0.512 | +0.041 | 0.040 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.452 → this 0.509 (+0.057 ± 0.014)
- inclusive (sensitivity), same thresholds: 0.372 → 0.421 (+0.049)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.843 | 0.819 | -0.024 | 122 |
| untagged | 0.474 | 0.531 | +0.057 | 2418 |
| background | 0.404 | 0.444 | +0.040 | 1874 |
| quiet | 0.110 | 0.126 | +0.016 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Headline +0.057 ± 0.014 (≈4 SD), first draw +0.063: two-draw mean 0.512, +0.060 over control and +0.032 over depth 10's two-draw mean (0.480).
Folds: 1_37 +0.161 (±0.062) reproduces; 1_29 +0.069 (±0.023), 1_143 +0.067 (±0.045, unsure), 1_150 +0.068 (±0.051, unsure, up both draws), 1_95 +0.035 (±0.023, unsure, up both draws), 1_114 +0.041 (unsure; depth 10 had it down both draws), 53 and 1_11 flat.
Tiers: untagged +0.057, background +0.040, quiet +0.016, loud -0.024 (122 frames; -0.048 in draw 1).

## Conclusion
Confirmed: fine-tuning layers 8-14 is +0.060 over layers 12-14 (two draws 0.515/0.509) and +0.032 over layers 10-14. Gain is in untagged and background tiers; loud slips slightly on small n. Hard folds lean up on both draws but each within its own SD. Next rung: depth 6.
