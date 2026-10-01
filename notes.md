# ps-bntrain-featdrop
## Hypothesis
Trainable tail BatchNorm (`ps-bntrain`, +0.027 / repeat +0.017) and channel dropout of the layer-11 map (`ps-featdrop`, +0.009, 1_150/1_95 leaned up) act on different things (BN adapts the statistics to the deployment mix, dropout regularises site-specific channels). With BN adapting, the tail may overfit more, so dropout may now pay. Control: `v4-ft-ps` (0.452, 30 epochs); also compare `ps-bntrain` (0.479) and `ps-bntrain-repeat` (0.469).

## Changes
Env knobs only (default off), in a copied `embedders/yamnet_trunk_depth12/embedder.py`: TRUNK_BN_TRAIN=1 and TRUNK_SPDROP=0.2. Everything else as v4-ft-ps: TRUNK_LR_BACKBONE=1e-5, TRUNK_LR_HEAD=2e-4, TRUNK_BATCH=1024, TRUNK_FP16=1, 30 epochs, CHUNK_FRAMES=48.
