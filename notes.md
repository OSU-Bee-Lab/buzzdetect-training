# ps-untied
## Hypothesis
v4-ft-ps runs the plain and octave-up views through one shared fine-tuned tail (TimeDistributed). The two views have different statistics (the up view is a pitch-shifted spectrum), so one tail must serve both. Giving each view its own tail (both from AudioSet weights, 2x tail parameters) lets each specialise; the linear head still concatenates the two pooled codes. Prediction: modest gain; risk of overfitting with 2x trainable parameters.

Control: `v4-ft-ps` (0.452), identical otherwise (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
`embedders/yamnet_trunk_depth12/embedder.py` (worktree copy): env knob `TRUNK_UNTIED=1` builds one tail per view and concatenates. Launch env: TRUNK_UNTIED=1.
