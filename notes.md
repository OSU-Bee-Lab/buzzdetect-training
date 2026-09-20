# trunk-ft-v3

## Hypothesis

Unfreezing YAMNet's last two separable-conv blocks (layers 13-14, ~1.6M params)
lets AudioSet filters specialise toward buzz. In earlier eras this was the best
unfreezing depth: `trunk-ft-1e5` was +0.046 at 9/11 folds (E2), and unfreezing
layer 12 as well (`unfreeze-more`, 12-14) was worse (frozen 0.216 -> 13-14 0.262
-> 12-14 0.229). Those are E2 numbers on old data and an old metric, so this is
a rerun under the current era (cv-medium-v3: fixed `--epochs`, `sensitivity_exclquiet`
at fpr 0.005, revised annotations), not a confirmation. The injunction on backbone
unfreezing was suspended by Luke on 2026-09-18.

Prediction: a modest gain at a small backbone LR, turning into overfit (hard folds
first) as the LR rises. With a fixed epoch budget and no early stopping, the
overfit end of the ladder is expected to look worse than it did under early
stopping.

## Changes

Ported by hand from `exp/trunk-ft` (a cherry-pick conflicted in 14 hunks against
today's `train.py`), not merged.

- `embedders/yamnet_trunk/` (shared tree): caches `layer12_pointwise_conv_relu`,
  (6,4,512) flattened to 12288-d float16. `build_head()` lifts layers 13-14 + GAP
  from `yamnet.keras`, BatchNorm frozen, then Dense. Per-variable LR multiplier
  for the backbone tail (`AdamMultiLR`).
- `03_train/train.py`: `_train_one` uses `embedder.build_head` if present and keeps
  the weighted BCE loss and label smoothing. Config comes from env vars, not CLI
  flags: `TRUNK_LR_BACKBONE`, `TRUNK_LR_HEAD` (2e-4), `TRUNK_BATCH`, `TRUNK_FP16`
  (float16 in memory, cast per batch).
- `_score_fold` and `surprisal.py` now run the model in batches of 1024. The whole
  fold in one call OOMed the 2.5 GB card on 12288-d input.
- `chain.sh`: the queue. `lite` smoke test, `medium` extraction, then six CVs.

## Design

All arms: `--set medium --embedder yamnet_trunk --translation general`, batch 1024,
head LR 2e-4, GPU. Batch 4096 OOMed in training on the 4 GB card, so batch was
dropped to 1024 (the old trunk-ft batch).

| run | backbone LR | epochs | other |
|---|---|---|---|
| `trunk3-frozen` | 0 | 60 | matched control (linear probe through the same pipeline) |
| `trunk3-ft-1e5` | 1e-5 | 60 | the old best LR |
| `trunk3-ft-3e5` | 3e-5 | 60 | |
| `trunk3-ft-1e4` | 1e-4 | 60 | |
| `trunk3-ft-1e5-e120` | 1e-5 | 120 | budget check |
| `trunk3-ft-3e5-do2` | 3e-5 | 60 | `--dropout 0.2` |

Matched control is `trunk3-frozen`, NOT `cv_baseline_v3`: batch, head LR, float16
cache and epoch budget all differ from the era anchor. `trunk3-frozen` vs
`cv-baseline-v3-refresh` separately shows what that pipeline change costs.
The 60-epoch budget is a guess (old best epochs were ~35 under early stopping;
400 epochs of fine-tuning would take too long). A different budget is a different
rule, so the 120-epoch run is not comparable to the 60-epoch ones without its
own control.

## GPU tuning (`tools_gpu_probe.py`, synthetic batch through the real head, GTX 1650)

| batch | ms/step | us/frame | peak MiB |
|---|---|---|---|
| 1024 | 40 | 39 | 2256 |
| 2048 | 82 | 40 | 2144 |
| 3072 | 124 | 41 | 901 |
| 4096 | 168 | 41 | 1103 |

Throughput is flat in batch size: the card is compute-bound at 1024, so a larger
batch buys nothing. Peak is set by cuDNN autotune workspace, not batch, and is
non-monotonic; this is why 4096 OOMed unpredictably in the real pipeline (dataset
cache plus autotune). ~40 us/frame means ~3 s/epoch on ~65k training frames, so a
60-epoch fold is ~3 min on GPU. Batch 1024 kept. Whole-fold scoring must be batched.

## Results

Metric: `sensitivity_exclquiet` at fpr 0.005, per-deployment mean over 8 rotating folds; ± is eval-sampling
SD only (training stochasticity is larger per fold). Matched control `trunk3-frozen` = 0.321.
`trunk3-frozen` vs `cv-baseline-v3-refresh` (0.329): -0.008 ± 0.006, so the pipeline change (batch, head LR,
fp16 cache, 60 epochs) costs nothing detectable.

| run | headline | delta vs `trunk3-frozen` |
|---|---|---|
| `trunk3-ft-1e5` | 0.398 | +0.077 ± 0.014 |
| `trunk3-ft-3e5` | 0.396 | +0.075 ± 0.015 |
| `trunk3-ft-1e5-e120` | 0.396 | +0.075 ± 0.015 |
| `trunk3-ft-3e5-do2` | 0.391 | +0.070 ± 0.013 |
| `trunk3-ft-1e4` | 0.386 | +0.065 ± 0.016 |

Per fold, `trunk3-ft-1e5` vs `trunk3-frozen` (delta ± SD): 1_29 +0.033 ± 0.034; 53 +0.130 ± 0.046;
1_11 (willard) +0.223 ± 0.047; 1_143 +0.070 ± 0.035; 1_150 +0.065 ± 0.048 (within SD); 1_95 +0.035 ± 0.025
(0.052 -> 0.087, still at floor); 1_37 +0.064 ± 0.035; 1_114 -0.008 ± 0.036 (flat).
Tiers: untagged +0.083, background +0.081, loud +0.024, quiet +0.030, faint 0 (12 frames).

Context (`yamnet_trunk_context`, 3 frames, 120 epochs, batch 512), 8/8 folds:

| run | headline | vs `trunk3-ft-1e5` |
|---|---|---|
| `trunkctx-frozen` | 0.371 | -0.027 ± 0.017 |
| `trunkctx-ft-1e5` | 0.400 | +0.002 ± 0.018 |
| `trunkctx-ft-3e5` | 0.334 | -0.064 ± 0.019 |

`trunkctx-ft-1e5` vs `trunkctx-frozen`: +0.029 ± 0.018, so fine-tuning helps context less than it helps the
plain trunk (+0.077). Context and fine-tuning do not stack. Against `trunk3-ft-1e5` on the hard folds:
1_95 +0.010 ± 0.031, 1_114 +0.051 ± 0.037, 1_150 -0.111 ± 0.051.

Budget (mean per-epoch val sens from the training callback, which reads lower than the headline, so use the
shape only): `trunk3-ft-1e5-e120` 0.297/0.306/0.319/0.327/0.327/0.323 at epochs 15/30/45/60/90/120, so 60
epochs is on the plateau. `trunkctx-ft-3e5` peaks by epoch ~23 (0.358) and decays to 0.273 by epoch 120: the
higher backbone LR overfits under a 120-epoch budget, which is why its CV is worse.

## Interpretation

Unfreezing layers 13-14 is a real gain of about +0.07 (about 5 SD), matching the E2 direction (+0.046) at a
larger size. Backbone LR from 1e-5 to 1e-4, 60 vs 120 epochs, and dropout 0.2 are indistinguishable at this
noise; 1e-4 is nominally lowest, consistent with the predicted overfit end. The prediction of a gain turning
into overfit was only weakly borne out on the plain trunk, and clearly on context at 3e-5 x 120 epochs.
The gain lands on willard and 53 and on untagged/background frames; the hard folds 1_95, 1_150 and 1_114 did
not move beyond noise, so this does not address the jet-flyover false positives at 1_95.

Deployment: the 5 plain-trunk models ship to buzzdetect. The parity check reports ~1e-3 logit drift with
agreement 1.0000 and exact parity at fixed 200 s length (chunk-boundary padding, shipped with `-y`). Context
models were not deployed: `yamnet_trunk_context.to_onnx` produces duplicate initializer names
(`head_const_fold_opt__N_folded_bias`, `onnx.checker`); skipped by Luke's decision since context does not stack.

## Conclusion

Trunk fine-tuning (layers 13-14) is a robust ~+0.07 on the current era; LR, epoch budget and dropout do not
matter within 1e-5..1e-4 and 60..120 epochs. Context does not stack with it. Not merged to main: the trunk
embedders live untracked in main's `embedders/`. Next lever, if any: what fine-tuning does for the hard
folds (1_95, 1_114, 1_150), which it did not fix.
