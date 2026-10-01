# ps-bntrain-gmp
## Hypothesis
Global max pooling concatenated with the average pool alone was neutral (`ps-gmp` 0.453 vs 0.452). With the tail BatchNorm trainable (`ps-bntrain` 0.479, repeat 0.469) the layer-11 statistics adapt to the deployment mix, so the max-pooled transient channel may now be usable. Control: `v4-ft-ps` (0.452); also compare `ps-bntrain` and `ps-bntrain-repeat`.

## Changes
Env knobs only: TRUNK_BN_TRAIN=1 and TRUNK_GMP=1, in the copied `embedders/yamnet_trunk_depth12/embedder.py` (knobs default off). Everything else as v4-ft-ps: LR backbone 1e-5, head 2e-4, batch 1024, FP16, 30 epochs, CHUNK_FRAMES=48.

## Results
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.499 | 0.508 | +0.009 | 0.036 | 32 |
| 53 | 0.517 | 0.560 | +0.043 | 0.033 | 28 |
| 1_11 | 0.575 | 0.537 | -0.038 | 0.020 | 26 |
| 1_143 | 0.559 | 0.550 | -0.009 | 0.028 | 22 |
| 1_150 | 0.312 | 0.324 | +0.012 | 0.050 | 21 |
| 1_95 | 0.201 | 0.212 | +0.011 | 0.019 | 46 |
| 1_37 | 0.480 | 0.534 | +0.054 | 0.043 | 14 |
| 1_114 | 0.471 | 0.352 | -0.119 | 0.052 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.452 → this 0.447 (-0.005 ± 0.013)
- inclusive (sensitivity), same thresholds: 0.372 → 0.370 (-0.002)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.843 | 0.820 | -0.023 | 122 |
| untagged | 0.474 | 0.471 | -0.003 | 2418 |
| background | 0.404 | 0.426 | +0.022 | 1874 |
| quiet | 0.110 | 0.114 | +0.004 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

## Conclusion
GMP on top of trainable tail BN: 0.447 vs 0.452 control (-0.005 +/- 0.013), below ps-bntrain (0.479) and its repeat (0.469). The BN gain vanished; the max-pooled channel is not usable even with adapted statistics. 1_114 fell 0.119 (+/- 0.052) while other folds were mixed; weak evidence, single draw. Dead end for GMP.
