# buzz-in-train
## Hypothesis
All buzz samples from the validate fold are moved into training. The model sees more buzz during learning, which may spur better buzz feature acquisition. Early stopping continues to function using the non-buzz classes remaining in validation — the assumption is that loss on those classes is a reasonable proxy for "good stopping point" even without buzz in val.

## Changes
- `03_train/train.py`: load buzz samples from validate fold and append to data_train; exclude buzz samples from data_val (keeping only non-buzz classes for early stopping)

## Results
- Baseline (with-dropout): 0.215, 0.236, 0.242, 0.241, 0.215  mean=0.229  median=0.236  95% CI=[0.216, 0.241]
- This experiment (buzz_in_train v1–v5): mean=0.204  median=0.189  std=0.042  95% CI=[0.151, 0.257]

CIs overlap substantially but mean is lower and variance is higher.

## Conclusion
Negative. Moving buzz to train and dropping it from validation hurts or is neutral at best. The early stopping proxy (non-buzz val classes) is apparently not well-calibrated for buzz — the model stops at a suboptimal point for the target class. Higher variance suggests the stopping criterion is less stable without buzz in val.
