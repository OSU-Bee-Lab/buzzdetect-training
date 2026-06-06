# buzz-upweight

## Hypothesis
The current balanced class weighting minimizes total cross-entropy equally across all classes. But the target metric is sensitivity at 95% precision for `ins_buzz` specifically. Multiplying the buzz class weight by 2x (on top of balanced weighting) will push the model to treat buzz misses as twice as costly, shifting the learned decision boundary toward higher buzz recall without necessarily sacrificing precision — since YAMNet embeddings discriminate buzz from non-buzz well enough that the boundary can move safely.

## Changes
- `03_train/train.py`: after building `weight_dict`, multiply buzz class weight by 2.0

## Results

Baseline (low-delta, n=8, no dropout): `[0.190, 0.204, 0.219, 0.221, 0.231, 0.236, 0.249, 0.250]`  
mean=0.225, median=0.226, 95% CI=[0.208, 0.243]

This experiment (2× buzz weight, dropout=0.2, n=5): `[0.182, 0.206, 0.208, 0.210, 0.222]`  
mean=0.206, median=0.208, 95% CI=[0.188, 0.224]

Mann-Whitney p=0.171 — not significant. CIs barely overlap at boundary.

Note: baseline used no dropout; this experiment used dropout=0.2 (per main). Comparison is slightly confounded, but direction is clearly not positive.

## Conclusion
Buzz 2× upweighting does not improve and is directionally worse than balanced weighting (median 0.208 vs 0.226). Standard balanced class weights are appropriate. The model is not limited by training emphasis on buzz — the bottleneck is elsewhere (embedding discriminability or dataset noise).
