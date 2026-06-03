# binary-translation

## Hypothesis
Collapsing all non-buzz classes into a single `background` class forces the linear probe to focus
entirely on the buzz/background boundary instead of wasting capacity on irrelevant distinctions.
Combined with eps=0.2 label smoothing and Dropout(0.3) on embeddings.

## Changes
- `translations/binary.csv`: ins_buzz → ins_buzz, all else → background (except mech_drone → ignore)
- `03_train/train.py`: Dropout(0.3) before Dense; label_smoothing=0.2

## Reproduction
- No external artifacts
- Train: `python 03_train/main.py --model exp_binary_v1 --set standard --embedder yamnet --translation binary --epochs 300`
- Stopped at epoch 58 (early stopping, patience=30)

## Results
- Baseline (exp_label_smooth02_v1): 0.2993
- This experiment (exp_binary_v1): 0.2844

## Interpretation
Worse by ~1.5pp. Two confounds: binary translation may have removed useful gradient signal
(fine-grained class distinctions teach better buzz/non-buzz separation), and Dropout(0.3) may
be too aggressive. Val loss (0.81) >> train loss (0.29) suggests poor generalization — consistent
with dropout masking useful signal on these dense embeddings.

## Conclusion
Binary translation + Dropout(0.3) does not improve over multi-class + eps=0.2. Next: test dropout
alone (general translation + eps=0.2 + Dropout(0.2)) to isolate the effect. Also consider that
the 11-class formulation may be providing beneficial auxiliary supervision.
