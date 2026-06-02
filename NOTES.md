# deeper-std

## Hypothesis
The existing `exp/deeper-head` experiment added a Dense(128, relu) hidden layer but trained on `lite`, getting 22.14% — worse than the `test_standard` baseline of 25.06%. However, `test_standard` used the much larger `standard` set. This experiment runs the same deeper head on `standard` to give a fair comparison. If the MLP head genuinely helps, more data should let it win.

## Evidence
- YAMNet embeddings (1024-dim) are generic audio representations; a single linear layer may not have enough capacity to isolate buzz-specific features.
- The `exp/deeper-head` result is confounded: lite < standard in data size. Can't conclude deeper head is worse from that comparison alone.
- MLP probing over pretrained audio embeddings typically outperforms linear probing (HEAR benchmark).

## Changes
- `03_train/train.py`: Add Dense(128, relu) before the output layer (same change as exp/deeper-head)
- Same `standard` set and `yamnet` embedder as the current best baseline (test_standard)
- All other hyperparameters unchanged
