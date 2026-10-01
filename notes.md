# ps-bntrain-depth10
## Hypothesis
ps-bntrain (trainable tail BatchNorm, 30 epochs) was -0.004..+? vs control at depth 12 and did not compound at 60 epochs; ps-depth10 (+0.021) fine-tunes layers 10-14. A deeper tail has more BN layers whose AudioSet moving stats mismatch the deployments, so adapting them may help more here. Prediction: small gain over ps-depth10 (0.473) at best; may be null like at depth 12.

Control: `v4-ft-ps` (0.452); also compare to ps-depth10 (0.473). Same config otherwise (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
`embedders/yamnet_trunk_depth10/embedder.py` (worktree copy): env knob `TRUNK_BN_TRAIN=1` makes tail BatchNorm layers trainable. Launch with TRUNK_BN_TRAIN=1, embedder yamnet_trunk_pitchshift_depth10.
