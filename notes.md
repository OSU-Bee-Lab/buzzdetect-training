# ps-d8-taps
## Hypothesis
The `ps-depth8` readout sees only YAMNet's last pooled map: layer 14, after layer 13's stride-2 has halved a 6x4 map to 3x2. The hard folds fail on confusers, not on missing buzz (`1_95` aircraft, `1_114` orthopteran trill, `1_37` "background"; `docs/training-proposer-facts.md`), and what separates a buzz from a jet or a trill may be mid-level time-frequency texture that the last two blocks were trained (on AudioSet) to abstract away. AVES middle layers were a confirmed gain in the frozen era (+0.026 to +0.032), and no run has given the fine-tuned YAMNet tail's own middle layers to the readout. So: let the class Dense read, per view, the global-average-pooled outputs of layers 10 and 12 (both 6x4x512, inside the trainable tail) next to layer 14's code (readout 2048 -> 4096 wide).

Expected signature of a real effect: headline above the matched control by more than ~2 SD of a delta (~0.03), with the gain in `untagged` / `loud` and on `1_95` / `1_114` / `1_37`. Falsifiers: headline within its SD of the control; or a gain that lives only in `1_29` / `53`'s `background` tier (another rich-fold lever).

This is not a context/averaging lever (same audio window, same frame), and not a new embedder: the cache is `yamnet_trunk_pitchshift_depth8`'s, unchanged.

Control: `ps-depth8-r3` (this batch's redraw from the same main commit), with `ps-depth8` (0.515) and `ps-depth8-repeat` (0.509) as the earlier draws.

## Changes
`03_train/train.py`: `TRUNK_TAPS=<layer>[,<layer>...]` rebuilds the `build_head()` model so the tail also outputs GAP of each named layer, concatenated before its own pooled code (`_add_taps`). Same tail weights, same per-variable learning rates, no dropout, no hidden layer. Unset: unchanged.

Run: `TRUNK_TAPS=layer10_pointwise_conv_relu,layer12_pointwise_conv_relu TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, `--embedder yamnet_trunk_pitchshift_depth8 --epochs 30`, set medium, translation general.
