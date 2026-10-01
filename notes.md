# ps-bntrain-e60
## Hypothesis
Trainable tail BatchNorm (`ps-bntrain`, +0.027 at 30 epochs) may compound with the longer 60-epoch budget (`v4-ft-ps-e60`, 0.468), since BN adaptation plus more steps lets the tail re-fit further. Control: `v4-ft-ps-e60` (matched 60 epochs, no BN training).

## Changes
`TRUNK_BN_TRAIN=1` knob only; 60 epochs, otherwise v4-ft-ps config.
