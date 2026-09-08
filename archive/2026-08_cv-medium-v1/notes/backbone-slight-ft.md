# backbone-slight-ft

## Hypothesis

The probe sits on a frozen AudioSet backbone whose last blocks were trained to
separate 521 everyday sound classes, none of which is an insect. Letting
YAMNet's last two separable-conv blocks (layers 13-14, ~1.6M params) move a
little — 1e-5 against the head's 2e-3, so ~200x slower — should let those
filters specialise toward buzz without the representation drifting far enough to
overfit eleven deployments' worth of data.

`yamnet-ft` tried exactly this before the rework (layers 13-14 at 1e-5) and lost
8.3pp, train 78% vs val 59%. That verdict comes from the retired corpus and the
fixed train/validate split, and `temporal-context` has already shown a verdict
from that era inverting outright under CV. The overfitting *mechanism* it
described is plausible either way, which is why this run keeps BatchNorm frozen
and the backbone LR small.

Prediction: if the old result was a data artifact, the buzz-specific filters win
and most folds move up. If it was real, the fine-tuned folds overfit and val
loss turns up early — which the per-fold `best_epoch` will show directly.

## Changes

Training on cached embeddings can't fine-tune anything: the cache *is* the
backbone's output. So the cut moves one block earlier.

- **`embedders/yamnet_trunk/`** — new embedder. Same YAMNet weights, same
  framing, but it caches `layer12_pointwise_conv_relu`, shaped (6, 4, 512),
  flattened to 12288 and stored float16. Everything below layer 13 stays frozen
  and is computed once, at extraction.
- **`embedders/yamnet_trunk/head.py`** — `build_head()` lifts layers 13-14 out
  of the loaded `yamnet.keras` (the layer objects themselves, so kernels and BN
  stats are the real ones, not a transcription) and stacks the usual
  GAP → Dropout(0.2) → Dense on top. With `lr_backbone=0` this is *exactly* the
  stock linear probe — verified numerically against the `yamnet` embedder:
  max abs difference 9.2e-4 on an embedding of scale 3.4, which is the float16
  cache quantisation and nothing else.
- **`AdamScaledLR`** — Adam with a per-variable LR multiplier, set in `build()`
  and applied in `update_step()`. Keras 3 has no parameter groups. Scaling the
  *gradient* instead would have been a no-op: Adam divides by its own RMS, so a
  constant factor on the gradient cancels exactly.
- BatchNorm in layers 13-14 is frozen in both arms. Its moving statistics are
  AudioSet's; the pre-rework "BatchNorm on input embeddings" run (-5.8pp) is
  what happens when a small set gets to rewrite them.
- **`03_train/train.py`** — `_train_one` asks the embedder for a head if it has
  one, otherwise builds the linear probe as before; `_to_tf` keeps float16
  embeddings float16 in the pipeline and casts a batch at a time (as float32 the
  medium set is ~3.6 GB, and `.cache()` would hold a second copy); `--batch` and
  `--lr-backbone` are new CLI flags.

### The forced confound, and the control

A 12288-d input cannot be trained at the default batch of 65568 (that one batch
is 3.2 GB of input tensor alone, on a 16 GB machine). Both arms run at
`--batch 4096`, i.e. ~17 steps an epoch where baseline gets ~2. That changes the
update regime, and per `stopping-rule-scale` in IDEAS.md the early-stopping rule
is sensitive to exactly that.

So the comparison that matters here is **not** against `cv-baseline`. It is
against a frozen control run through the identical trunk pipeline at the
identical batch size, differing only in `--lr-backbone`:

| run | model | `--lr-backbone` |
|---|---|---|
| control | `trunk_frozen` | 0 |
| treatment | `trunk_ft1e5` | 1e-5 |

The control against `cv-baseline` separately measures what the batch change
alone did, which is worth knowing regardless of how the fine-tuning lands.

## Results

## Conclusion
