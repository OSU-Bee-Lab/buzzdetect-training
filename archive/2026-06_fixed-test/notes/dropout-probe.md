<!-- harvested 2026-09-25 from exp/dropout-probe (2188ba0); pinned at refs/archive/dropout-probe -->

<!-- experiments/dropout-probe/notes.md -->
# dropout-probe

## Hypothesis
Dropout(0.2) on YAMNet embeddings before the linear probe adds regularization that is
otherwise absent. Combined with proven eps=0.2 label smoothing, general 11-class translation.
Isolates the dropout effect from the binary-translation experiment (binary-translation used
Dropout(0.3) and binary translation simultaneously).

## Changes
- `03_train/train.py`: Dropout(0.2) before Dense; label_smoothing=0.2
- Translation: general (unchanged)

## Reproduction
- No external artifacts
- Train: `python 03_train/main.py --model exp_dropout_v1 --set standard --embedder yamnet --translation general --epochs 300`
- Stopped at epoch 44 (early stopping, patience=30)

## Results
- Baseline (exp_label_smooth02_v1): 0.2993 sensitivity@95% precision
- This experiment (exp_dropout_v1): 0.3175 sensitivity@95% precision

## Interpretation
New best result. +1.8pp over prior best. The binary-translation experiment showed that
collapsing to 2 classes hurts (0.2844), suggesting the 11-class supervision provides a
useful inductive bias. Dropout(0.2) alone atop that multi-class signal is beneficial.
The previous experiment's failure was the translation change, not the dropout.

## Conclusion
Dropout(0.2) + eps=0.2 + general translation is the new best configuration.
Next: try dropout=0.3 or 0.4 with general translation to see if more regularization
continues to help; also consider mixup in embedding space.
