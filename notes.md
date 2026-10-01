# ps-bntrain-depth10
## Hypothesis
ps-bntrain (trainable tail BatchNorm, 30 epochs) was -0.004..+? vs control at depth 12 and did not compound at 60 epochs; ps-depth10 (+0.021) fine-tunes layers 10-14. A deeper tail has more BN layers whose AudioSet moving stats mismatch the deployments, so adapting them may help more here. Prediction: small gain over ps-depth10 (0.473) at best; may be null like at depth 12.

Control: `v4-ft-ps` (0.452); also compare to ps-depth10 (0.473). Same config otherwise (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
`embedders/yamnet_trunk_depth10/embedder.py` (worktree copy): env knob `TRUNK_BN_TRAIN=1` makes tail BatchNorm layers trainable. Launch with TRUNK_BN_TRAIN=1, embedder yamnet_trunk_pitchshift_depth10.

## Results
30 epochs, 8 folds. sens_exclquiet 0.455 vs control v4-ft-ps 0.452 (+0.003 ± 0.014); vs ps-depth10 0.473 (-0.018). Inclusive 0.371 vs 0.372. Per fold: 1_29 -0.119, 1_114 -0.210 (large drops), 1_143/1_150/1_37 +0.08..+0.09. Tiers: background -0.055, loud -0.026, quiet +0.023.

## Interpretation
Trainable BN at depth 10 erases the depth-10 gain (0.473 -> 0.455), same pattern as depth 12 (null). Two folds collapse (1_29, 1_114) while others gain; the net is within noise of control. BN adaptation does not compound with the deeper tail.

## Conclusion
Null / negative vs ps-depth10. Keep frozen BN at depth 10. 
