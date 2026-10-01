# ps-ema
## Hypothesis
Each fold's headline-relevant weights are the final-epoch weights of a noisy 30-epoch fine-tune (training SD ~0.011-0.018 per fold). An exponential moving average of the weights (momentum 0.998, ~500 steps, a few epochs) is a structural noise reducer: it averages the late trajectory without changing the budget or any data. Prediction: small gain from lower variance, most visible on thin folds; no hard-fold lift expected.

Control: `v4-ft-ps` (0.452), identical otherwise (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
`embedders/yamnet_trunk_depth12/embedder.py` (worktree copy): env knob `TRUNK_EMA=momentum` turns on Keras optimizer weight EMA; fit() swaps EMA weights in at the end. Launch env: TRUNK_EMA=0.998.
