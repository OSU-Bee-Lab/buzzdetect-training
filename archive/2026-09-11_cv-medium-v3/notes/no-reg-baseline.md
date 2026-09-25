# no-reg-baseline

## Hypothesis
Establish a clean multi-run (n=5) true baseline: linear probe on YAMNet embeddings with zero regularization (no dropout, no label smoothing, no L2). All prior multi-run experiments used LS=0.2; this gives an unambiguous reference point.

## Changes
- `03_train/train.py`: remove Dropout(0.2) layer; set label_smoothing=0.0

## Results
- This experiment (exp_no_reg v1–v5): 0.183, 0.187, 0.192, 0.198, 0.207  mean=0.193  median=0.192  std=0.010  95% CI=[0.181, 0.205]
- with-dropout (dropout=0.2 + LS=0.2): mean=0.229  95% CI=[0.216, 0.241]

No regularization baseline sits 3.6pp below the current default config. Very low variance (std=0.010) — convergence is stable but the model doesn't generalize as well.

## Conclusion
True zero-regularization baseline is 0.193 — substantially below current defaults. Confirms dropout + label smoothing together are contributing ~3.6pp over bare probe. Useful as reference for isolating individual regularizer contributions.
