# deeper-head

## Hypothesis
The current classifier is a single Dense layer (logistic regression) over YAMNet's 1024-dim embeddings. YAMNet embeddings encode many audio concepts; buzz-relevant features are likely entangled with irrelevant ones in a nonlinear way. Adding one hidden layer (Dense 128, ReLU) before the output should allow the model to learn nonlinear combinations and improve sensitivity without sacrificing precision.

## Changes
- `03_train/train.py`: Added `Dense(128, activation='relu')` before the output layer
- Set: `lite`, embedder: `yamnet`, translation: `general`

## Reproduction
No external artifacts. Run from the worktree:
```bash
conda run -n buzzdetect-train python 03_train/main.py \
  --model exp_deeper_head_lite --set lite --embedder yamnet \
  --translation general --epochs 300
```

## Results

Note: no single-Dense baseline exists for the `lite` set with test results, so a direct same-set comparison isn't available. The best available reference is `test_standard` (single Dense, `standard` set), which is a larger dataset.

Metric | test_standard (single Dense, standard) | exp_deeper_head_lite (Dense 128, lite)
---|---|---
Sensitivity @ 95% precision | 25.07% | 22.25%
Sensitivity @ 90% precision | 27.63% | 26.47%
Precision @ 80% sensitivity | 25.63% | 24.51%

The deeper head is worse across all metrics, though the comparison is confounded by dataset size (lite < standard). The `exp/deeper-std` experiment (same MLP head, standard set) confirmed the pattern holds even with equal data: 19.35% vs 25.07% — the MLP head hurts more, not less, on standard.

## Conclusion
MLP head is consistently worse than linear probe on YAMNet embeddings at the high-precision operating point. Do not revisit MLP head variants unless there's a specific reason to expect different behavior (e.g., much larger dataset, regularisation changes, or a different embedder with less-linear structure).
