# ps-noise
## Hypothesis
Cutout and channel dropout (structured masking of the layer-11 map) were null. Additive Gaussian noise is the unstructured regulariser: sigma = 0.1 x the batch std of the map, training only, smooths the tail's decision surface against deployment-specific feature shifts without removing any cue. Prediction: null to small gain; the background tier is the one to watch since noise could mask quiet buzzes.

Control: `v4-ft-ps` (0.452), identical otherwise (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
`embedders/yamnet_trunk_depth12/embedder.py` (worktree copy): env knob `TRUNK_NOISE=sigma` adds a training-only Gaussian noise layer after the reshape. Default off. Launch env: TRUNK_NOISE=0.1.
