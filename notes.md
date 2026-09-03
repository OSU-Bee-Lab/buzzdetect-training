# unfreeze-more

Follow-up to `exp/trunk-ft`. Branched from `exp/trunk-ft` @ 38ee546 (not main —
main has none of the trunk fine-tune plumbing).

## Hypothesis

`trunk-ft-1e5` unfroze YAMNet's last two separable-conv blocks (layers 13-14,
~1.6M params) at backbone LR 1e-5 and gained **+0.046 vs a matched frozen
control** (9/11 folds up) — the largest structural gain in the CV log. The
open dose-response question: does unfreezing *more* of the backbone help
further, or does it tip into overfit on `medium` (5662 buzz frames)?

This arm unfreezes **layers 12-14** (adds layer 12's depthwise + pointwise
convs, ~0.27M params, to the trainable set). Backbone LR held at 1e-5, head at
2e-4 — one thing changed vs `trunk-ft-1e5`.

Prediction: given the pre-rework `yamnet-ft` overfit signature (train 78 / val
59) inverted once already under CV, a modest further gain is plausible; but the
`medium` set may be too small to feed 3 unfrozen blocks and the extra capacity
could just overfit (earlier `best_epoch`, higher `best_val_loss`).

### Caveat carried from the user (2026-09-02)

The user's concern: the unfreezing gain may only widen on a **larger** dataset,
and `medium` (like `large`) is still mid-annotation. `large` (5x frame density,
`framehop_prop 0.2`) exists but is too slow for a LOOP cycle and its
`folds_sx.csv` is not comparable to the CV log (different negative population).

The one comparable data point from the partial `exp/large-trunk-ft` run
(layers 13-14, backbone 1e-5, on `large`):

| fold | medium baseline | medium trunk-ft-1e5 (13-14) | large trunk-ft-1e5 (13-14) |
|---|---|---|---|
| `Diel Drivers/2026-04-08/1_150` | 0.014 | 0.068 | 0.114 (best_epoch 1 — shaky) |

Monotone climb baseline → 13-14 unfreeze → 5x density, consistent with the
user's report that higher frame density synergizes with unfreezing — but it's
one collapsed-early fold, not evidence. If this arm (12-14 on `medium`) is
neutral-or-negative, that does **not** rule out 12-14 helping on `large`; it
only says `medium` can't feed it. Note that explicitly in the conclusion.

## Changes

- **`embedders/yamnet_trunk11/`** — copy of `yamnet_trunk`, cut one block
  earlier: caches `layer11_pointwise_conv_relu` ((6,4,512) → 12288 float16,
  identical on-disk size to `yamnet_trunk` — channels don't grow until layer
  13's pointwise conv, inside the trainable tail). `build_head()` lifts layers
  **12-14** + GAP out of `yamnet.keras`; BN in 12-14 stays frozen. Requires a
  fresh `medium` extraction (reuses the `sr16000_fl0.96` framed-audio cache, so
  no re-framing / no source-audio reads).
- No `03_train` changes — `--lr-backbone` / `--batch` / `build_head` wiring all
  inherited from `exp/trunk-ft`.

Smoke test (`tmp/smoke.py`): head builds, 8 trainable vars (layer12/13/14
depthwise+pointwise kernels + dense kernel/bias, BN frozen), fits one step,
`save(include_optimizer=True)` → `load(compile=False)` round-trips to 0.0 diff.

## Runs

`medium`, 11 rotating folds, `general` translation, `--batch 1024`,
`--lr-backbone 1e-5 --lr-head 2e-4`, `clipnorm 1.0`, patience 50, min_delta
0.002, epochs cap 400. All CPU (`CUDA_VISIBLE_DEVICES=""`) — 12288-d @ batch
1024 OOMs the 4 GB GPU, and CPU ≈ GPU on this box for stage 3.

| stage | cmd | status |
|---|---|---|
| 2. extract | `02_set/main.py --set medium --embedder yamnet_trunk11 --workers 2` | running |
| 3. train `unfreeze_more_1e5` | `03_train/main.py --name unfreeze_more_1e5 --set medium --embedder yamnet_trunk11 --translation general --lr-backbone 1e-5 --lr-head 2e-4 --batch 1024 -y` | pending |

Comparators (from `.local/worktrees/trunk-ft/models/`):
`trunk_frozen` 0.216 (frozen control), `trunk_ft_1e5` 0.262 (13-14 unfreeze).

## Results

_pending_

## Conclusion

_pending_
