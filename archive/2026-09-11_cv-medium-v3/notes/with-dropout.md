# with-dropout

## Hypothesis
Dropout(0.2) on input embeddings, with standard class weighting and all other low-delta
defaults (label_smoothing=0.2, lr=0.002, min_delta=0.002, patience=50, epochs=400), is
neutral or beneficial vs the no-dropout low-delta baseline (0.226 mean).

The low-delta baseline had no dropout. Dropout was added to main's defaults (commit 43a959e)
before this experiment. buzz-upweight confounded dropout with 2x class weight, so this
experiment isolates dropout alone.

## Changes
None vs current main — this is a baseline check for the current default config.

## Results
- Baseline (low-delta, no dropout): 8 runs, mean=0.224, 95% CI=[0.207, 0.242]
- This experiment (with_dropout v1–v5): 0.215, 0.224, 0.230, 0.233, 0.242  mean=0.229  median=0.230  95% CI=[0.216, 0.241]
CIs overlap heavily. Dropout is neutral — statistically indistinguishable from no-dropout.

## Conclusion
Dropout(0.2) is neutral vs no-dropout. Current defaults are fine and stable. The bottleneck
is not regularization. Remove dropout from the defaults consideration; it doesn't matter.
