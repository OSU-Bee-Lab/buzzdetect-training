# unfreeze-one

Based on `exp/trunk-ft` @ 38ee546 (not main), because the comparator and the
whole trunk pipeline live there.

## Hypothesis

The depth dose-response for trunk fine-tuning has a measured middle and a
measured deep end, but no shallow end:

| unfroze | experiment | sens@fpr0.005 |
|---|---|---|
| nothing (frozen control) | `trunk-frozen` | 0.216 |
| layers 13-14 | `trunk-ft-1e5` | **0.262** |
| layers 12-14 | `unfreeze-more` | 0.229 |

Going deeper turned the curve over (median `best_epoch` 35 -> 25, the predicted
overfit signature), and `IDEAS/lora-adapter` now records 13-14 as "settled" on
`medium`. But settled between two points that both have more capacity than the
untried one: **layer 14 alone**. If the turnover at 12-14 is capacity-driven
overfitting on ~5.6k buzz frames, the optimum may sit shallower than 13-14, not
at it. 14-only has 1.06M trainable params vs 13-14's 1.59M — a third less,
still 12x the frozen probe's head.

Prediction: 14-only lands between frozen (0.216) and 13-14 (0.262). If it lands
at or above 0.262, the loop has been paying for layer 13 for nothing and the
whole "unfreeze more" axis should be re-read as a regularization axis.

## Changes

`embedders/yamnet_trunk/embedder.py` only, ~10 lines: `build_head` now consults
`BUZZDETECT_TRUNK_FT_BLOCKS` (default `13,14`) and sets `layer.trainable` per
block. Blocks not listed stay frozen but remain in the graph, computed from the
same layer-12 cache — **so this needs no re-extraction**, unlike `unfreeze-more`
(which had to cache layer 11 to reach deeper). With the default the graph, the
trainable-variable set and the optimizer's backbone multiplier are bit-identical
to `trunk-ft-1e5`.

Verified before launching, via `build_head(2, lr_backbone=1e-5, lr_head=2e-4)`:

| env | trainable tail layers | trainable params |
|---|---|---|
| `13,14` (default) | layer13/14 depthwise+pointwise conv | 1,588,738 |
| `14` | layer14 depthwise+pointwise conv | 1,059,842 |

BatchNorm stays frozen in both, as before.

## Run

```
BUZZDETECT_TRUNK_FT_BLOCKS=14 03_train/main.py --name unfreeze_one --set medium \
  --embedder yamnet_trunk --translation general \
  --lr-backbone 1e-5 --lr-head 2e-4 --batch 1024 --verbose -y
```

Everything else matches `trunk-ft-1e5`: min_delta 0.002, patience 50, epochs 400,
`general` translation, CPU-only.

## Results

_pending_

## Conclusion

_pending_
