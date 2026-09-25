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
| 2. extract | `02_set/main.py --set medium --embedder yamnet_trunk11 --workers 2` | done |
| 3. train `unfreeze_more_1e5` | `03_train/main.py --name unfreeze_more_1e5 --set medium --embedder yamnet_trunk11 --translation general --lr-backbone 1e-5 --lr-head 2e-4 --batch 1024 -y` | done |

Comparators (from `.local/worktrees/trunk-ft/models/`):
`trunk_frozen` 0.216 (frozen control), `trunk_ft_1e5` 0.262 (13-14 unfreeze).

## Results

`models/unfreeze_more_1e5/folds_sx.csv`, `total` row: **mean sens@fpr0.005 = 0.229**.

Paired against the two arms from `exp/trunk-ft`, same folds:

| fold | frozen | ft 13-14 | ft 12-14 (this) | vs 13-14 | buzz | neg |
|---|---|---|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 0.441 | 0.469 | 0.472 | +0.003 | 2144 | 24 |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 0.370 | 0.432 | 0.437 | +0.005 | 1031 | 18 |
| Lily Adam/willard/2024-08-07/1_11 | 0.204 | 0.318 | 0.286 | -0.032 | 305 | 22 |
| Lily Adam/wooster/2024-07-26/1_143 | 0.284 | 0.337 | 0.295 | -0.042 | 190 | 22 |
| Luke - Diel Drivers/2026-04-08/1_150 | 0.048 | 0.068 | 0.082 | +0.014 | 146 | 24 |
| Luke - Diel Drivers/2026-05-06/1_95 | 0.035 | 0.048 | 0.046 | -0.002 | 433 | 30 |
| Luke - Various/2025-06-23/1_23 | 0.337 | 0.400 | 0.246 | -0.154 | 35 | **1** |
| Luke - Various/2025-07-03/1_37 | 0.303 | 0.293 | 0.336 | +0.043 | 307 | 22 |
| Luke - Various/2025-08-05/31 | 0.135 | 0.185 | 0.118 | -0.067 | 411 | **2** |
| Luke - Various/2025-08-12/1_114 | 0.194 | 0.316 | 0.136 | -0.180 | 361 | 17 |
| Luke - Various/2025-08-27/48 | 0.027 | 0.020 | 0.070 | +0.050 | 299 | 6 |

- mean sens@fpr0.005: frozen control 0.216 → **12-14 0.229** → 13-14 **0.262**
- vs 13-14 (`trunk_ft_1e5`): **-0.033**, 5 folds up / 6 down
- vs the frozen control (`trunk_frozen`): **+0.013**, 8 folds up / 3 down

**Which folds moved.** The two largest losses vs 13-14 are the two folds whose
0.5% FPR threshold rests on **1 and 2 negative frames** (`1_23`, `2025-08-05/31`)
— those rows are not readable as capability in either direction. Strip them and
the remaining nine are 5 up / 4 down for a mean of -0.010, i.e. flat. The one
substantial readable loss is `2025-08-12/1_114` (-0.180, 361 buzz, 17 neg), and
the one substantial readable gain is `2025-08-27/48` (+0.050, a near-chance fold).

**Overfit signature, as predicted.** Median `best_epoch` falls
36 (frozen) → 35 (13-14) → **25 (12-14)**, and the two folds that lost most
stopped at epoch 5 (`1_114`) and 80 (`1_23`). The extra 0.27M trainable params
buy earlier convergence, not a better optimum.

## Conclusion

**Negative-to-inconclusive: unfreezing layer 12 on top of 13-14 does not help
on `medium`.** The dose-response turns over — frozen 0.216 → 13-14 0.262 →
12-14 0.229 — so 13-14 remains the operating point, and there is no reason to
unfreeze further on this set. The -0.033 vs 13-14 is above the ~0.017 measured
noise floor, but the folds are split 5/6 and half the gap comes from two rows
resting on one or two negative frames; read the direction (no gain), not the
size.

The `+0.013` over the frozen control is the honest floor on this arm: 12-14 is
still better than not fine-tuning at all, just worse than fine-tuning less.

**This does not close the `large`-set question.** Per the caveat above, the
prediction was that extra unfrozen capacity needs more data, and `medium` has
5662 buzz frames. The earlier stopping seen here is exactly what "too little
data for the capacity" looks like, so 12-14 on `large` (5x frame density) is
still open — this run says only that `medium` cannot feed it.
