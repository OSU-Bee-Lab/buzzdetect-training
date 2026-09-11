# l2-regularize

## Hypothesis
L2 regularization (λ=1e-4) on the Dense layer kernel will shrink irrelevant YAMNet embedding dimensions and improve generalization at the 95% precision operating point. The linear probe currently has no weight decay — on a 1024-d input with small training data, it may be overfitting to spurious embedding dimensions.

## Changes
- `03_train/train.py`: add `kernel_regularizer=tf.keras.regularizers.l2(1e-4)` to the Dense layer

## Results
- Baseline (with-dropout): 0.215, 0.241, 0.220, 0.233, 0.235  mean=0.229  95% CI=[0.216, 0.241]
- This experiment (exp_l2_reg v1–v5): 0.180, 0.216, 0.223, 0.230, 0.244  mean=0.219  median=0.223  95% CI=[0.189, 0.249]

CIs overlap but lower bound regresses substantially (0.189 vs 0.216). Mean drops 1pp.

## Conclusion
Negative: L2(1e-4) on the Dense kernel hurts. Dropout(0.2) + label_smoothing=0.2 already provide sufficient regularization; explicit weight decay over-regularizes. Dead end at this λ value.
