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
evidence without changing the frame population, the unfreeze depth, or the
head's output width (still 1024-d, so nothing downstream moves). Comparator is
`trunk_ft_1e5` (0.262), not `cv-baseline` — same embedder family, same
unfreeze depth (13-14), same LRs/batch; the only change is what feeds the
Dropout->Dense head.

Prediction: most folds move up, `willard` (highest fraction of short/isolated
buzz events, the fold this mechanism is explicitly aimed at) moves most.

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
  --embedder yamnet_trunk --translation general --lr-backbone 1e-5 \
  --lr-head 2e-4 --batch 1024 --verbose -y
```

Matches `trunk_ft_1e5` exactly except the pooling change above: min_delta
0.002, patience 50, epochs_max 400, `general` translation (this branch
predates the `general_v1` freeze), CPU-only.

## Results

(fill in when the CV finishes — see HANDOFF_subframe-head.md)

## Conclusion

(fill in when the CV finishes)
