# yamnet-ft

## Hypothesis
Extracting at YAMNet layer 12 (last 512-filter block, output shape 6×4×512) and fine-tuning
the remaining two separable conv blocks (layers 13–14, ~1024-filter) with a very low learning rate
will allow the backbone to adapt its high-level features toward buzz discrimination, while keeping
the lower-level acoustic features intact. Two-phase training: (1) warm up the Dense head with the
YAMNet tail frozen; (2) unfreeze the YAMNet tail (BN frozen) and fine-tune with LR=1e-5.
Baseline: low-delta (0.224 mean, CI [0.207, 0.242]).

## Changes
- New embedder `yamnet_l12`: extracts at `layer12_pointwise_conv_relu`, emitting (6, 4, 512) per frame
- Training model: Input(6,4,512) → layers 13–14 (unfrozen phase 2, BN frozen) → GAP → Dropout(0.2) → Dense
- Phase 1: Adam(lr=0.002), EarlyStopping(patience=10)
- Phase 2: Adam(lr=1e-5), EarlyStopping(patience=50), BN layers in tail frozen

## Results
- Baseline (with-dropout): mean=0.229  95% CI=[0.216, 0.241]
- This experiment (yamnet_ft_v1–v2, only 2 runs): 0.136, 0.155  mean=0.146

Note: only 2 runs completed (v3–v5 aborted after observing strong overfitting pattern).
Train accuracy reached ~78% vs val ~59% by epoch 125 of phase 2. CIs are meaningless at n=2,
but both values are far below baseline.

## Conclusion
Clear negative: fine-tuning YAMNet layers 13–14 hurts substantially (−8.3pp vs with-dropout).
Overfitting is the likely cause — backbone adapts to training distribution, degrades on test.
The small dataset (~few hundred buzz frames) cannot support backbone adaptation even at LR=1e-5.
Frozen embeddings + linear probe is the right inductive bias for this data size.
