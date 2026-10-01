# ps-depth10-repeat
## Hypothesis
ps-depth10 (fine-tune layers 10-14) gave +0.021 +/- 0.013 over v4-ft-ps: under the MDE and one draw. The loop's rule is to confirm a gain with one repeat run before building on it. Prediction: if real, the repeat lands near +0.02 with the same signature (1_37 up, 1_114 down); if it is training noise it regresses to ~0.

Control: `v4-ft-ps` (0.452) and `ps-depth10` (0.473, first draw), identical config (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
None. Same code and cache (`yamnet_trunk_pitchshift_depth10`); only the training draw differs.
