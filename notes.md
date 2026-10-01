# ps-bntrain-gmp
## Hypothesis
Global max pooling concatenated with the average pool alone was neutral (`ps-gmp` 0.453 vs 0.452). With the tail BatchNorm trainable (`ps-bntrain` 0.479, repeat 0.469) the layer-11 statistics adapt to the deployment mix, so the max-pooled transient channel may now be usable. Control: `v4-ft-ps` (0.452); also compare `ps-bntrain` and `ps-bntrain-repeat`.

## Changes
Env knobs only: TRUNK_BN_TRAIN=1 and TRUNK_GMP=1, in the copied `embedders/yamnet_trunk_depth12/embedder.py` (knobs default off). Everything else as v4-ft-ps: LR backbone 1e-5, head 2e-4, batch 1024, FP16, 30 epochs, CHUNK_FRAMES=48.
