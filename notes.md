# ps-d8-foldbal
## Hypothesis
The headline is a plain mean over deployments, but the training loss is a plain mean over frames, and deployments differ in size by about 100x (medium: 1 to 120 units of frames per fold, median 17; the eight rotating folds are among the largest and `1_95` is the largest of all). A rotation's pool is therefore dominated by a handful of big deployments, and the fine-tuned tail can spend its capacity on their backgrounds. Group-balanced ERM (each domain reweighted toward equal total weight) is the standard strong baseline for generalising to an unseen domain. So: scale each training frame's loss by its fold's `n_frames ** -0.5`, normalised to mean 1 over the pool; p = 0.5 halves the spread in log terms without handing a 1-unit fold 100x per-frame weight.

This reweights the class loss only. It adds no site head and no term on the pooled code, so it is not one of the ruled-out site-adversarial / site-confusion shapes. Class weights (`build_weights`) and the validation fold's loss are unchanged, so val_loss curves stay comparable.

Expected signature of a real effect: headline above the matched control by more than ~2 SD of a delta (~0.03), with gains on the hard folds (`1_95`, `1_150`, `1_114`) and in `untagged` / `loud`. Falsifiers: headline within its SD of the control; or the big-fold-rich tiers (`background`, `1_29` / `53`) falling while nothing else rises (the weights just took data away).

Control: `ps-depth8-r3` (this batch's redraw from the same main commit), with `ps-depth8` (0.515) and `ps-depth8-repeat` (0.509) as the earlier draws.

## Changes
`03_train/train.py`: `TRUNK_FOLDBAL=<p>` puts each training frame's fold weight in a trailing target column (`_targets`, `_load_data`) and the loss multiplies the per-frame class loss by it. Unset: unchanged.

Run: `TRUNK_FOLDBAL=0.5 TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, `--embedder yamnet_trunk_pitchshift_depth8 --epochs 30`, set medium, translation general.

`[foldbal]` line per rotation (48 training folds each): frame weight min 0.55-0.57, median 1.90-1.94, max 6.35-6.49. The weights span ~11.5x, so fold frame counts in a pool span ~130x (p = 0.5); that is the measured form of the Hypothesis's rough "1 to 120 units". Mean weight over frames is 1, so the median fold's frames count ~1.9x and the biggest folds' ~0.55x.

## Results
`buzz logit SD` per fold 0.53-1.11 (`1_150` lowest at 0.53, below the usual 0.65-1.2 but nowhere near a collapse); all folds trained the full 30 epochs, best epoch 30.

Against `ps-depth8-r3` (same-day control, same main commit):

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.445 | -0.041 | 0.044 | 32 |
| 53 | 0.550 | 0.520 | -0.030 | 0.034 | 28 |
| 1_11 | 0.573 | 0.585 | +0.012 | 0.026 | 26 |
| 1_143 | 0.586 | 0.577 | -0.009 | 0.027 | 22 |
| 1_150 | 0.435 | 0.389 | -0.046 | 0.054 | 21 |
| 1_95 | 0.284 | 0.267 | -0.017 | 0.025 | 46 |
| 1_37 | 0.648 | 0.579 | -0.069 | 0.061 | 14 |
| 1_114 | 0.414 | 0.289 | -0.125 | 0.053 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.456 (-0.041 ± 0.015)
- inclusive (sensitivity), same thresholds: 0.411 → 0.373 (-0.038)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.820 | -0.036 | 122 |
| untagged | 0.524 | 0.480 | -0.044 | 2418 |
| background | 0.406 | 0.374 | -0.032 | 1874 |
| quiet | 0.130 | 0.107 | -0.023 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only; training stochasticity is not in it.

Against `ps-depth8-repeat` (0.509): -0.053 ± 0.016. Against the lowest of the three plain draws it is still -0.041.

Headline: -0.041 ± 0.015 eval SD, about -2.4 SD with training stochasticity added (~0.017 combined): a negative, not a null. Per fold, only `1_114` moves beyond its own SD (-0.125 ± 0.053, a hard fold, ~2 SD even with training noise); `1_37` -0.069 ± 0.061, `1_150` -0.046 ± 0.054, `1_29` -0.041 ± 0.044 and `53` -0.030 ± 0.034 are each within or about 1 SD, unsure alone; `1_95`, `1_143`, `1_11` unmoved. No fold rose beyond its SD. Every scored tier fell by a similar amount (loud -0.036, untagged -0.044, background -0.032): the loss is not confined to the big-fold `background` tier, so the second falsifier's specific shape (only big-fold tiers fall) is not what happened; the first one's stronger form did (headline below control, hard folds down).

## Conclusion
Fold-balanced loss at p = 0.5 hurts: 0.456 against 0.497 (-0.041 ± 0.015), with the hard fold `1_114` down -0.125 ± 0.053 and every tier down 0.03-0.04. Down-weighting the big deployments' frames to ~0.55x costs more than up-weighting small ones buys; frame-mean ERM stays. One draw at one p; a milder p is a hyperparameter probe and not obviously worth a CV given the uniform direction.
