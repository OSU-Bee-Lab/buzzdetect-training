# ps-cutout
## Hypothesis
Channel dropout on the layer-11 map (ps-featdrop) was flat. Deployments differ in where in the time-frequency grid the buzz energy sits (6 time rows x 4 frequency columns of the layer-11 map); masking one random row or column per view forces the tail to classify from partial evidence, a SpecAugment-style structural regulariser on spatial rather than channel structure. Prediction: small gain at best, possibly on 1_114/1_150 where context pieces matter; risk of washing out the transient (hard-fold prior).

Control: `v4-ft-ps` (0.452), identical otherwise (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
`embedders/yamnet_trunk_depth12/embedder.py` (worktree copy): env knob `TRUNK_CUTOUT=p` adds a training-only layer after the reshape: per sample and per view, with probability p zero one random time row (half the time) or one random frequency column. Default off. Launch env: TRUNK_CUTOUT=0.5.

## Results
Mean sens@fpr0.005 (exclquiet): 0.446 vs control v4-ft-ps 0.452 (-0.006 +/- 0.012). No fold missed fpr 0.005. Inclusive 0.372 -> 0.365. Per fold: 1_37 +0.098 +/- 0.037 (14 events), 1_29 -0.044, 53 -0.054 +/- 0.055, 1_11 -0.031, 1_95 -0.038 +/- 0.019 (hard fold, worse), 1_143/1_150/1_114 ~0. Tiers: loud +0.009, background -0.068 (1874 frames), quiet flat.

## Interpretation
Mean is flat, inside the noise floor (headline delta SD ~0.014). The 1_37 gain is weak evidence on 14 events and is offset by losses elsewhere; the hard fold 1_95 moved the wrong way. Background-tier loss hints that masking a row/column removes the partial cues that quieter buzzes rely on, consistent with the transient-washout risk.

## Conclusion
Null: spatial cutout does not help, like channel dropout. Not pursued.
