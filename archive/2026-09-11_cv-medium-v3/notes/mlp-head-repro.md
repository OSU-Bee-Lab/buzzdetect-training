# mlp-head-repro

## Hypothesis
The MLP head (Dense 128 → Dense N_classes) underperformed in deeper-std, but that experiment predated stable training methodology (null commit, no multi-run averaging, wrong set). With current defaults (low-delta, 5 runs, label smoothing 0.2, dropout 0.2), MLP may perform comparably or better than the linear probe. A single hidden layer is the minimum nonlinearity to test whether the buzz manifold in YAMNet space is linearly separable or not.

## Changes
- Added `Dense(128, activation='relu')` between `Dropout(0.2)` and the output Dense layer in `03_train/train.py`
- Architecture: Input(1024) → Dropout(0.2) → Dense(128, relu) → Dense(N_classes)
- All other defaults unchanged (label_smoothing=0.2, lr=0.002, patience=50, min_delta=0.002)

## Results
- Baseline (with-dropout): 0.215, 0.223, 0.229, 0.236, 0.242  mean=0.229  median=0.229  95% CI=[0.216, 0.241]
- This experiment (mlp_head v1–v5): 0.195, 0.200, 0.200, 0.203, 0.206  mean=0.201  median=0.200  95% CI=[0.196, 0.206]

CIs do not overlap (0.196–0.206 vs 0.216–0.241). MLP head is clearly worse.

## Conclusion
Negative: adding Dense(128, relu) drops mean by 2.8pp with no CI overlap. Loss curves show train and val descending together with no divergence — the model is not overfitting. The hidden layer simply doesn't find generalizable nonlinear structure in YAMNet embeddings. The linear probe is the better inductive bias: buzz is already linearly accessible in this space, and extra capacity doesn't add transferable signal. Dead end for architecture complexity in this direction.
