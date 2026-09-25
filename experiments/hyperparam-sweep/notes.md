# hyperparam-sweep

## Hypothesis
Sequential independent sweeps of dropout rate, label smoothing epsilon, and learning rate.
Prior experiments established Dropout(0.2) + label_smooth=0.2 + lr=0.002 as best known config (0.3175 @ 95% precision),
but each parameter was tuned separately in different sessions with different baselines. Re-sweeping all
values in the current combined config may reveal a different optimum due to interactions.

## Changes
- `03_train/train.py`: added `dropout_rate`, `label_smoothing`, and `learning_rate` args to `train_model()`
- `03_train/main.py`: thread new args through CLI
- Sweep order: dropout → label_smooth → lr (each sequential, using best from prior stage)

## Reproduction
No external artifacts.

## Results
TBD

## Conclusion
TBD
