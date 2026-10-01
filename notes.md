# ps-cutout
## Hypothesis
Channel dropout on the layer-11 map (ps-featdrop) was flat. Deployments differ in where in the time-frequency grid the buzz energy sits (6 time rows x 4 frequency columns of the layer-11 map); masking one random row or column per view forces the tail to classify from partial evidence, a SpecAugment-style structural regulariser on spatial rather than channel structure. Prediction: small gain at best, possibly on 1_114/1_150 where context pieces matter; risk of washing out the transient (hard-fold prior).

Control: `v4-ft-ps` (0.452), identical otherwise (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
`embedders/yamnet_trunk_depth12/embedder.py` (worktree copy): env knob `TRUNK_CUTOUT=p` adds a training-only layer after the reshape: per sample and per view, with probability p zero one random time row (half the time) or one random frequency column. Default off. Launch env: TRUNK_CUTOUT=0.5.
