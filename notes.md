# ps-bntrain-e60
## Hypothesis
Trainable tail BatchNorm (`ps-bntrain`, +0.027 at 30 epochs) may compound with the longer 60-epoch budget (`v4-ft-ps-e60`, 0.468), since BN adaptation plus more steps lets the tail re-fit further. Control: `v4-ft-ps-e60` (matched 60 epochs, no BN training).

## Changes
`TRUNK_BN_TRAIN=1` knob only; 60 epochs, otherwise v4-ft-ps config.

## Results
60-epoch CV vs `v4-ft-ps-e60` (0.468): mean sens@fpr0.005 (exclquiet) 0.457, delta -0.011 +/- 0.014 (eval-sampling SD). No fold missed the FPR target.

| fold | baseline | this | delta | ± SD |
|---|---|---|---|---|
| 1_29 | 0.543 | 0.456 | -0.087 | 0.035 |
| 53 | 0.542 | 0.559 | +0.017 | 0.035 |
| 1_11 | 0.554 | 0.585 | +0.031 | 0.017 |
| 1_143 | 0.631 | 0.613 | -0.018 | 0.040 |
| 1_150 | 0.296 | 0.312 | +0.016 | 0.039 |
| 1_95 | 0.229 | 0.233 | +0.004 | 0.029 |
| 1_37 | 0.516 | 0.537 | +0.021 | 0.038 |
| 1_114 | 0.430 | 0.358 | -0.072 | 0.066 |

Tiers: loud +0.040, untagged -0.003, background -0.051, quiet -0.003.

## Conclusion
Trainable BN does not compound with 60 epochs: -0.011 +/- 0.014, within noise (headline training SD ~0.005, delta SD ~0.014), versus +0.017/+0.027 at 30 epochs. The gain at 30 epochs looks like a faster-convergence effect that the longer budget absorbs. Hard folds 1_150/1_95 are flat; 1_29 and 1_114 (trill) drop. Not adopted at 60 epochs.
