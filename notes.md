# ps-d8-taps
## Hypothesis
The `ps-depth8` readout sees only YAMNet's last pooled map: layer 14, after layer 13's stride-2 has halved a 6x4 map to 3x2. The hard folds fail on confusers, not on missing buzz (`1_95` aircraft, `1_114` orthopteran trill, `1_37` "background"; `docs/training-proposer-facts.md`), and what separates a buzz from a jet or a trill may be mid-level time-frequency texture that the last two blocks were trained (on AudioSet) to abstract away. AVES middle layers were a confirmed gain in the frozen era (+0.026 to +0.032), and no run has given the fine-tuned YAMNet tail's own middle layers to the readout. So: let the class Dense read, per view, the global-average-pooled outputs of layers 10 and 12 (both 6x4x512, inside the trainable tail) next to layer 14's code (readout 2048 -> 4096 wide).

Expected signature of a real effect: headline above the matched control by more than ~2 SD of a delta (~0.03), with the gain in `untagged` / `loud` and on `1_95` / `1_114` / `1_37`. Falsifiers: headline within its SD of the control; or a gain that lives only in `1_29` / `53`'s `background` tier (another rich-fold lever).

This is not a context/averaging lever (same audio window, same frame), and not a new embedder: the cache is `yamnet_trunk_pitchshift_depth8`'s, unchanged.

Control: `ps-depth8-r3` (this batch's redraw from the same main commit), with `ps-depth8` (0.515) and `ps-depth8-repeat` (0.509) as the earlier draws.

## Changes
`03_train/train.py`: `TRUNK_TAPS=<layer>[,<layer>...]` rebuilds the `build_head()` model so the tail also outputs GAP of each named layer, concatenated before its own pooled code (`_add_taps`). Same tail weights, same per-variable learning rates, no dropout, no hidden layer. Unset: unchanged.

Run: `TRUNK_TAPS=layer10_pointwise_conv_relu,layer12_pointwise_conv_relu TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, `--embedder yamnet_trunk_pitchshift_depth8 --epochs 30`, set medium, translation general.

## Results
All 8 folds trained (`buzz logit SD` 0.57-1.12, fold for fold the same as the control's). `[taps] readout width 4096` on every rotation. Against `ps-depth8-r3` (same main commit, same day):

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.484 | -0.002 | 0.033 | 32 |
| 53 | 0.550 | 0.586 | +0.036 | 0.027 | 28 |
| 1_11 | 0.573 | 0.580 | +0.007 | 0.023 | 26 |
| 1_143 | 0.586 | 0.559 | -0.027 | 0.031 | 22 |
| 1_150 | 0.435 | 0.352 | -0.083 | 0.052 | 21 |
| 1_95 | 0.284 | 0.222 | -0.062 | 0.022 | 46 |
| 1_37 | 0.648 | 0.672 | +0.024 | 0.039 | 14 |
| 1_114 | 0.414 | 0.392 | -0.022 | 0.036 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.481 (-0.016 ± 0.012)
- inclusive (sensitivity), same thresholds: 0.411 → 0.397 (-0.014)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.820 | -0.036 | 122 |
| untagged | 0.524 | 0.499 | -0.025 | 2418 |
| background | 0.406 | 0.440 | +0.034 | 1874 |
| quiet | 0.130 | 0.125 | -0.005 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Against `ps-depth8-repeat`: 0.509 → 0.481 (-0.028 ± 0.016); against the three plain draws' mean (0.507), -0.026.

Interpretation. The headline is -0.016 ± 0.012 against the matched control, about one full SD (~0.014 with training stochasticity) on the wrong side: the first falsifier is met, there is no gain. Whether it is a loss is unsure: 0.481 is below all three plain draws (0.497, 0.509, 0.515), which is weak evidence of a small cost, not more.

Folds: the two that moved past their eval SD are the hard ones, down: `1_95` -0.062 ± 0.022 and `1_150` -0.083 ± 0.052. `ps-depth8-redraw` showed two identical runs of this recipe differing by up to ~0.1 on a fold, so each is unsure as an effect; but against `ps-depth8-repeat` both are also below (-0.014, -0.028), so nothing here says the taps help the confuser folds, which was the prediction. `1_114` -0.022 ± 0.036 and `1_37` +0.024 ± 0.039 are within their SDs. `53` +0.036 ± 0.027 is the only fold up past its SD.

Tiers: the predicted signature was `untagged` / `loud` up; they went down (-0.025, -0.036 on 122 frames) and `background` went up (+0.034, `53`). That is the opposite of the hypothesis and the pattern of the second falsifier (a rich-fold lever), within a null headline.

## Conclusion
Null, leaning negative: giving the readout GAP of layers 10 and 12 next to layer 14 scored 0.481 against 0.497 (-0.016 ± 0.012), with the hard folds `1_95` and `1_150` down rather than up and `untagged` / `loud` down. Mid-level taps on a linear readout do not help the confuser folds; the fine-tuned tail's last code already carries what a linear head can use from them. Not tried: taps from the frozen layers below the tail (7-9), or a non-linear readout over them.
