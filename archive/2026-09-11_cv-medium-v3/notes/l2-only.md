# l2-only

## Hypothesis
Test L2(1e-4) as the sole regularizer on the linear probe — no dropout, no label smoothing. Previous l2-regularize experiment combined L2 with dropout+LS, which may have over-regularized. This isolates L2's contribution against the true zero-regularization baseline (no-reg-baseline).

## Changes
- `03_train/train.py`: remove Dropout(0.2) layer; set label_smoothing=0.0; add kernel_regularizer=l2(1e-4) to Dense layer

## Results
- This experiment (exp_l2_only v1–v5): 0.170, 0.187, 0.194, 0.195, 0.205  mean=0.190  median=0.194  std=0.013  95% CI=[0.174, 0.206]
- no-reg-baseline (no dropout, no LS): mean=0.193  95% CI=[0.181, 0.205]
- with-dropout (dropout=0.2 + LS=0.2): mean=0.229  95% CI=[0.216, 0.241]

L2-only is identical to no regularization (CIs fully overlap). Current defaults outscore both by ~4pp.

## Conclusion
L2 kernel regularization provides no benefit as a standalone regularizer — essentially equivalent to an unregularized probe. The ~3.6pp gain in the current defaults comes from dropout + label smoothing, not from anything L2 can replicate. L2 regularization is a dead end for this architecture.
