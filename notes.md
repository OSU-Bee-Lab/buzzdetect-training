# plain-d8
## Hypothesis
The octave-up view was this era's largest single lever at depth12 (`v4-ft-ps` +0.077 over plain `v4-ft`), and it was never re-measured once the trainable tail grew to depth8 (+0.06 over depth12). Two questions, one run:

1. **Is the up view still needed at depth8?** Seven trainable blocks may re-form low-frequency features the up view supplied. Prediction: plain depth8 (`yamnet_trunk_depth8`, single view) lands >= 0.03 below `ps-depth8-r3` (0.497), i.e. the up view still carries a real share. Falsifier of the up view's value: within 0.02.
2. **Does a cross-recipe member decorrelate an ensemble better than a same-recipe draw?** Same-recipe ensembles saturate at three (`ps-d8-ens-same5` = ens3, 0.558). If plain-d8's negatives decorrelate more from ps-depth8's than ps draws do from each other, the logit mean of {plain-d8, two ps-depth8 draws} beats the best same-recipe triple from those draws. Prediction: plain-d8 + ps-depth8 + ps-depth8-repeat >= ps-d8-ens3 (0.558) + 0.02. Falsifier: at or below ens3.

Control: `ps-depth8-r3` (same tail, LR, epochs, batch; differs only in the view). Ensembles via `exp/ps-d8-ensemble`'s `tools_exp/ensemble_predictions.py`.

## Changes
`03_train/train.py`: `TRUNK_ONLY_VIEW=<v>` trains on one view of a multi-view trunk cache. The model is the embedder's own single-view head (`n_ctx` 1: Reshape -> depth8 tail -> Dense), fed view v's 12288 columns cropped out of the cached row with standard Keras layers (Reshape, Cropping1D). So rows, labels, frames and thresholds are the control's exactly; no new cache. Run: ps-depth8-r3's recipe on `yamnet_trunk_pitchshift_depth8` with `TRUNK_ONLY_VIEW=0` (plain view). Chain: `exp/binary-d8:chain_b41.sh`.

## Results
Every fold's `buzz logit SD` is in range (0.44 at 1_150 to 1.14 at 1_29).

### 1. plain depth8 vs `ps-depth8-r3`
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.450 | -0.036 | 0.031 | 32 |
| 53 | 0.550 | 0.541 | -0.009 | 0.036 | 28 |
| 1_11 | 0.573 | 0.528 | -0.045 | 0.032 | 26 |
| 1_143 | 0.586 | 0.587 | +0.001 | 0.047 | 22 |
| 1_150 | 0.435 | 0.380 | -0.055 | 0.052 | 21 |
| 1_95 | 0.284 | 0.233 | -0.051 | 0.025 | 46 |
| 1_37 | 0.648 | 0.559 | -0.089 | 0.032 | 14 |
| 1_114 | 0.414 | 0.069 | -0.345 | 0.048 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.418 (-0.079 ± 0.014)
- inclusive (sensitivity), same thresholds: 0.411 → 0.338 (-0.073)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.804 | -0.052 | 122 |
| untagged | 0.524 | 0.441 | -0.083 | 2418 |
| background | 0.406 | 0.391 | -0.015 | 1874 |
| quiet | 0.130 | 0.103 | -0.027 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

Headline -0.079 ± 0.014 (full SD with training noise ~0.016): the up view still carries a large share at depth8, about the same as it did at depth12 (+0.077). Prediction 1 (>= 0.03 below) holds.

Per fold, 1_114 carries it: -0.345 ± 0.048, from 0.414 to 0.069. That is the trill fold; the octave-up view is what separates trill from buzz there, and seven trainable blocks do not re-form it. Without 1_114 the other seven average -0.041, still a loss: 1_37 -0.089 ± 0.032 and 1_95 -0.051 ± 0.025 beyond their SDs; 1_150 -0.055 ± 0.052, 1_11 -0.045 ± 0.032 and 1_29 -0.036 ± 0.031 at about 1 SD (unsure individually, consistent in aggregate); 53 and 1_143 flat. Tiers: untagged -0.083, loud -0.052; background barely moved (-0.015).

### 2. Cross-recipe ensembles (logit mean, `tools_exp/ensemble_predictions.py`)
Same-recipe triple ps-d8-ens3 = {ps-depth8 (a), ps-depth8-repeat (b), ps-depth8-r3 (c)} = 0.558. Each row swaps one ps draw for plain-d8, and also compares against the ps pair plain-d8 joined:

| ensemble | headline | vs ps-d8-ens3 (0.558) | ps pair alone | vs pair | 1_114 vs ens3 |
|---|---|---|---|---|---|
| plain + a + b | 0.534 | -0.024 ± 0.009 | 0.556 | -0.022 ± 0.010 | -0.050 ± 0.039 |
| plain + a + c | 0.526 | -0.032 ± 0.009 | 0.544 | -0.018 ± 0.009 | -0.106 ± 0.037 |
| plain + b + c | 0.532 | -0.026 ± 0.009 | 0.538 | -0.006 ± 0.009 | -0.087 ± 0.041 |

Prediction 2 (>= 0.578) falsified: every cross-recipe triple is below ens3 by 0.024-0.032, and adding plain-d8 to a two-draw ps pair lowers it (-0.006 to -0.022) rather than raising it. The cost is again at 1_114 (plain's collapse drags the mean's logits on trill negatives) plus 1_95 (-0.027 to -0.038 vs ens3). The one positive is 1_150 (+0.002 to +0.030 vs ens3; +0.011 to +0.058 vs the pair, within ~1.4 SD at best): weak evidence that plain-d8 decorrelates there. Ensemble predictions are in `models/plain-d8-ens-*` and `models/ps-pair-*` (folds_sx.csv tracked).

## Conclusion
1. The up view is still needed at depth8: plain depth8 is -0.079 ± 0.014 vs ps-depth8-r3, almost entirely the trill fold 1_114 (-0.345 ± 0.048) with a smaller consistent loss elsewhere (-0.041 mean over the other seven). Trainable blocks do not replace the octave-up view.
2. A plain-view member does not decorrelate usefully: swapping it for a ps draw costs -0.024 to -0.032 vs ens3, and it lowers the two-draw ps pair it joins. A weaker member's failure mode (1_114) dominates the mean; ensemble diversity has to come from members that are individually as strong. Weak 1_150 gain only.
