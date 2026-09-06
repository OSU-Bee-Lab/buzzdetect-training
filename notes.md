# subframe-head

Based on `exp/trunk-ft` @ 38ee546 (not main) — the comparator (`trunk_ft_1e5`)
and the whole trunk fine-tune pipeline live there.

## Hypothesis

`yamnet_trunk` caches `layer12_pointwise_conv_relu` at shape (6, 4, 512) — 6
time steps and 4 frequency bands inside each 0.96s frame. Layers 13-14 stride
that to (3, 2, 1024), and `GlobalAveragePooling2D` then averages all six
remaining spatial positions away before the head ever sees them. For a buzz
that occupies a fraction of a 960ms frame, mean pooling divides its evidence
across time steps that may be silent, and does the same across the two
remaining frequency bands even though buzz is narrowband.

Replacing GAP with **max over the time axis, then mean over the frequency
axis** (IDEAS.md's `subframe-head`, option 1) should recover some of that
evidence without changing the frame population or the head's output width
(still 1024-d, so nothing downstream moves).

**Backbone stays frozen (`lr_backbone=0`) — layers 13-14 are not
fine-tuned here.** The only trainable weights are the usual Dropout->Dense
head; pooling itself has no parameters. This is deliberately not stacked on
top of the fine-tune: keeping the backbone frozen makes each pooling variant a
cheap, fast-iterating CV (same regime as `trunk_frozen`), so several pooling
options (this one, log-sum-exp, keep-frequency) can be tried in sequence
before spending a slow fine-tuned run on whichever wins. Comparator is
`trunk_frozen` (0.216), not `trunk_ft_1e5` and not `cv-baseline` — same trunk
pipeline, same frozen 13-14, same LR/batch regime; the only change is what
feeds the head.

Prediction: most folds move up, `willard` (highest fraction of short/isolated
buzz events, the fold this mechanism is explicitly aimed at) moves most. If
this wins clearly frozen, *then* it's worth re-testing stacked on the
fine-tune (against `trunk_ft_1e5`).

## Changes

`embedders/yamnet_trunk/embedder.py` only:
- `build_head()` now cuts the tail before `GlobalAveragePooling2D` (via
  `full.get_layer(_GAP_LAYER).input`) instead of at its output.
- New `TimeMaxFreqMean` layer (registered `keras.saving.register_keras_serializable`,
  built lazily like `AdamMultiLR`) does
  `reduce_mean(reduce_max(x, axis=time), axis=freq)` on the (batch,3,2,1024)
  tail output, giving the same 1024-d vector shape GAP used to produce.
- Everything else — which layers are trainable (13-14 only), BatchNorm frozen,
  the optimizer, the Dropout->Dense head, `n_embeddings` (12288) — is
  unchanged. No re-extraction; embeddings symlinked from `exp/trunk-ft`'s cache.

**Caught by `tools/`-style smoke test before spending a CV**: a first version
used `keras.layers.Lambda(lambda t: ...)`, which builds and trains fine but
fails `keras.saving.load_model(..., compile=False)` — the exact path
`write_model_py.py`'s inference wrapper uses — because keras refuses to
deserialize a raw Python lambda by default (`ValueError: ... disallowed by
default ... enable_unsafe_deserialization`). Replaced with a registered
`keras.layers.Layer` subclass, which builds, fits one step, and round-trips
through `model.save()`/`load_model()` cleanly (verified on dummy data before
launching the CV).

## Run

```
CUDA_VISIBLE_DEVICES="" 03_train/main.py --name subframe_head --set medium \
  --embedder yamnet_trunk --translation general --lr-backbone 0 \
  --lr-head 2e-4 --batch 1024 --verbose -y
```

Matches `trunk_frozen` exactly except the pooling change above: min_delta
0.002, patience 50, epochs_max 400, `general` translation (this branch
predates the `general_v1` freeze), CPU-only.

(A first launch used `--lr-backbone 1e-5`, i.e. stacked on the fine-tune —
wrong call, reverted before any fold finished. Backbone is frozen here.)

## Results

| fold | trunk_frozen sens@fpr0.005 | subframe_head | delta | val frames |
|---|---|---|---|---|
| Various Opportunistic/2025-08-05/31 | 0.135 | 0.060 | -0.075 | 942 |
| Various Opportunistic/2025-07-03/1_37 | 0.303 | 0.277 | -0.026 | 4715 |
| Diel Drivers/2026-04-08/1_150 | 0.048 | 0.034 | -0.014 | 4947 |
| wooster/2024-07-26/1_143 | 0.284 | 0.276 | -0.008 | 4708 |
| Diel Drivers/2026-05-06/1_95 | 0.035 | 0.028 | -0.007 | 6628 |
| Fit+Fast/2023_R3_Marysville/53 | 0.370 | 0.367 | -0.003 | 4712 |
| Various Opportunistic/2025-06-23/1_23 | 0.337 | 0.343 | +0.006 | 315 |
| JamesU - MustardBumbler/1_29 | 0.441 | 0.449 | +0.008 | 6984 |
| Various Opportunistic/2025-08-12/1_114 | 0.194 | 0.202 | +0.008 | 3768 |
| willard/2024-08-07/1_11 | 0.204 | 0.214 | +0.010 | 4730 |
| Various Opportunistic/2025-08-27/48 | 0.027 | 0.047 | +0.020 | 1571 |

- mean sens@fpr0.005: `trunk_frozen` 0.216 -> `subframe_head` 0.209 (delta -0.007)
- 5 folds up, 6 down — split, and the mean delta (-0.007) sits inside the
  noise floor (~0.014 headline, ~0.017 median per-fold; `noise-floor-cv`).
- `willard` — the named test case for this mechanism (highest fraction of
  short/isolated buzz events) — moved up, but only +0.010, well inside noise
  and not the standout mover the hypothesis predicted. The two biggest movers
  (`Various Opportunistic/31` at -0.075, `.../48` at +0.020) are both thin
  folds (942 and 1571 val frames) known to swing hard on their own, not
  folds this mechanism targets.
- `best_epoch` stayed in `trunk_frozen`'s range (10-88 per fold), no
  systematic early/late shift, so the pooling change isn't obviously
  reshaping the loss landscape — it's just not helping.

## Conclusion

Time-max/freq-mean pooling over the trunk's (6,4,1024) tail does not clear
the noise floor against `trunk_frozen`, and the fold that motivated the
hypothesis (`willard`) didn't move meaningfully. Negative/inconclusive:
mean-pool GAP evidence-dilution is not the bottleneck, at least not one this
pooling swap recovers. Not worth re-testing stacked on the fine-tune per the
hypothesis's own stopping condition ("if this wins clearly frozen, *then*
retest on the fine-tune") — it did not win clearly frozen.
