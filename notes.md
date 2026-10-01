# ps-bntrain-featdrop
## Hypothesis
Trainable tail BatchNorm (`ps-bntrain`, +0.027 / repeat +0.017) and channel dropout of the layer-11 map (`ps-featdrop`, +0.009, 1_150/1_95 leaned up) act on different things (BN adapts the statistics to the deployment mix, dropout regularises site-specific channels). With BN adapting, the tail may overfit more, so dropout may now pay. Control: `v4-ft-ps` (0.452, 30 epochs); also compare `ps-bntrain` (0.479) and `ps-bntrain-repeat` (0.469).

## Changes
Env knobs only (default off), in a copied `embedders/yamnet_trunk_depth12/embedder.py`: TRUNK_BN_TRAIN=1 and TRUNK_SPDROP=0.2. Everything else as v4-ft-ps: TRUNK_LR_BACKBONE=1e-5, TRUNK_LR_HEAD=2e-4, TRUNK_BATCH=1024, TRUNK_FP16=1, 30 epochs, CHUNK_FRAMES=48.

## Results
Mean sens@fpr0.005 (exclquiet): baseline 0.452 -> 0.448 (-0.004 +/- 0.014); inclusive 0.372 -> 0.368. No folds missed fpr.
Per fold (delta +/- SD): 1_29 -0.079/0.058, 53 +0.063/0.037, 1_11 -0.026/0.028, 1_143 -0.024/0.031, 1_150 +0.031/0.046, 1_95 -0.024/0.018, 1_37 +0.077/0.040, 1_114 -0.050/0.041.
Tiers: loud 0.843->0.804, untagged 0.474->0.479, background 0.404->0.378, quiet 0.110->0.108.

## Conclusion
Adding channel dropout (0.2) on top of trainable tail BN erases the BN gain: 0.448 vs control 0.452, versus 0.479 / 0.469 for BN alone. Within noise of control, below BN-only. No compounding; dropout does not pay with BN adapting. Not adopted.
