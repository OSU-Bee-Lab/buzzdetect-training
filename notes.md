# ps-noise
## Hypothesis
Cutout and channel dropout (structured masking of the layer-11 map) were null. Additive Gaussian noise is the unstructured regulariser: sigma = 0.1 x the batch std of the map, training only, smooths the tail's decision surface against deployment-specific feature shifts without removing any cue. Prediction: null to small gain; the background tier is the one to watch since noise could mask quiet buzzes.

Control: `v4-ft-ps` (0.452), identical otherwise (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
`embedders/yamnet_trunk_depth12/embedder.py` (worktree copy): env knob `TRUNK_NOISE=sigma` adds a training-only Gaussian noise layer after the reshape. Default off. Launch env: TRUNK_NOISE=0.1.

## Results
30 epochs, 8 folds. sens_exclquiet 0.461 vs control v4-ft-ps 0.452 (+0.009 ± 0.012); inclusive 0.372 -> 0.381. No fold missed fpr 0.005. Per fold: 1_29 -0.088 ± 0.040, 1_150 +0.068 ± 0.045, 1_37 +0.063 ± 0.032, 1_114 +0.049 ± 0.038, 1_95 +0.031 ± 0.022, 53/1_11/1_143 within SD. Tiers: background -0.076 (1874 frames), untagged +0.017, loud +0.009, quiet +0.008.

## Interpretation
Headline delta is inside the noise floor (headline delta SD ~0.014): null. Hard folds 1_150 and 1_95 moved up slightly (weak, near SD), 1_29 down. The background tier fell again (-0.076), as with cutout (-0.068): perturbing the layer-11 map costs quieter buzzes whatever the form.

## Conclusion
Null: Gaussian noise at 0.1 sigma is not a useful regulariser. Not pursued; input-space perturbations of the layer-11 map consistently hurt the background tier.
