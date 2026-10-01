# ps-bntrain-repeat
## Hypothesis
`ps-bntrain` (trainable tail BatchNorm) gave 0.479 vs `v4-ft-ps` 0.452 (+0.027, ~1.9 SD with training noise), the best pool-experiment headline of the era but unconfirmed. A repeat draw with identical config should land near +0.027 if real; near +0 if it was training noise. Control: `v4-ft-ps` (0.452) and the first draw `ps-bntrain` (0.479).

## Changes
Only the `TRUNK_BN_TRAIN` env knob (tail BN trainable when backbone LR > 0) copied into the worktree's `embedders/yamnet_trunk_depth12/embedder.py`. Launch: same as v4-ft-ps plus TRUNK_BN_TRAIN=1, 30 epochs.
