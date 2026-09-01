# trunk-ft

A multi-run investigation (LOOP.md sanctioned by the user for one session):
does unfreezing YAMNet's last blocks help on the CV metric, does augmentation
help once it's unfrozen, and is any further variant worth it.

## Hypothesis

The probe sits on a frozen AudioSet backbone whose last blocks separate 521
everyday sound classes, none an insect. Letting YAMNet's last two separable-conv
blocks (layers 13-14, ~1.6M params) move should let those filters specialise
toward buzz. `yamnet-ft` tried this pre-rework (layers 13-14, lr 1e-5) and lost
8.3pp with a clear overfit signature (train 78% / val 59%); the user reports a
later, unlogged CV-era attempt was "mildly beneficial". `temporal-context`
already inverted a pre-rework verdict under CV, and the current set is larger,
so the old negative is a lead, not a wall.

Prediction: if the old result was a data/eval artifact, buzz-specialised filters
win and most folds move up vs the frozen control. If it was real, the
fine-tuned folds overfit and val_loss turns up early — visible in per-fold
`best_epoch`.

## Design

Training on cached 1024-d GAP embeddings cannot fine-tune anything — the cache
*is* the backbone output. So the cut moves one block earlier.

- **`embedders/yamnet_trunk/`** — caches `layer12_pointwise_conv_relu`, (6,4,512)
  flattened to 12288, float16. Same YAMNet weights, same framing; reuses the
  existing `sr16000_fl0.96` framed-audio cache (identical `audio_cache_key`), so
  no re-framing.
- **`build_head(n_classes, lr_backbone, lr_head)`** lifts layers 13-14 + GAP out
  of the loaded `yamnet.keras` (real layer objects, real weights) and stacks
  Dropout(0.2) → Dense. `lr_backbone == 0` freezes 13-14 → numerically the stock
  linear probe (verified: max abs diff 1.06e-3 vs the `yamnet` embedder, all
  float16 cache quantisation). `lr_backbone > 0` makes their conv kernels
  trainable; **BatchNorm in 13-14 stays frozen either way** (moving stats are
  AudioSet's; letting a small set rewrite them cost -5.8pp pre-rework).
- **`03_train`** — `_train_one` asks the embedder for a head if it has one;
  `_to_tf` keeps float16 in the pipeline and casts per batch (float32 the medium
  pool is ~3.4 GB and `.cache()` would double it); `--batch`, `--lr-backbone`,
  `--lr-head` are new CLI flags. Optimizer is a single Adam + `clipnorm=1.0`
  (NaN insurance; this pipeline is NaN-prone per std-convergence).

### The forced confound and the control

12288-d cannot train at the default full batch (65568) in 2.4 GB of VRAM. All
arms run `--batch 4096` (~16 steps/epoch vs baseline's ~2), which changes the
update regime and, per `stopping-rule-scale`, the early-stopping rule is
sensitive to exactly that. So the comparison that matters is **not** vs
`cv-baseline`:

| run | `--lr-backbone` | `--lr-head` | note |
|---|---|---|---|
| `trunk-frozen`  | 0    | 2e-4 | control — linear probe through the same pipeline/batch/LR |
| `trunk-ft`      | 2e-4 | 2e-4 | treatment — 13-14 trainable, uniform LR |

`trunk-frozen` vs `cv-baseline` separately measures what the batch+LR+float16
change alone cost. `trunk-ft` vs `trunk-frozen` is the effect of unfreezing,
one thing changed.

Uniform LR (not the 200x-slower-backbone of the old run) keeps it a single
plain Adam and a fully serialisable functional model. A genuinely differential
rate is a later variant if the uniform one shows signal.

## Runs

All on `medium`, 11 rotating folds, `general` translation, `--batch 1024`
(12288-d cannot fit a larger batch in 2.4 GB VRAM), `--lr-head 2e-4`,
`clipnorm 1.0`, patience 50, epochs cap 400. Per-fold OOM between rotations was
routine on this GPU; a wrapper restarted `main.py` (which resumes from disk)
until `folds_sx.csv` existed.

| run | lr_backbone | total sens@fpr0.005 | vs frozen | vs cv-baseline |
|---|---|---|---|---|
| `cv-baseline` (`yamnet_medium_general`, cached GAP, full-batch, lr 2e-3) | — frozen | 0.199 | — | — |
| **`trunk_frozen`** (control: same trunk pipeline, 13-14 frozen) | 0 | **0.216** | — | +0.017 |
| **`trunk_ft_1e5`** (unfreeze 13-14) | 1e-5 | **0.262** | **+0.046** | +0.063 |
| **`trunk_ft_5e5`** | 5e-5 | **0.257** | +0.041 | +0.058 |
| `trunk_ft_1e5_aug` (+ within-fold noise 0.05) | 1e-5 | _see below_ | | |

Also run on `lite` as a smoke test only (not comparable): frozen 0.166,
uniform-lr-2e-4 fine-tune 0.150 (shipped best_epoch 6 — uniform 2e-4 on the
backbone overfits hard, which is why the medium runs use a differential rate).

### trunk_frozen vs cv-baseline

Per fold the two track each other closely (|Δ| < 0.02 on 6/11); trunk_frozen
nudges up on ~7/11 for +0.017 overall. Batch 1024 + lr 2e-4 vs full-batch +
lr 2e-3, plus the float16 trunk cache — within seed noise (std-convergence put
same-config rerun noise at 0.013-0.022). Frozen trunk+tail reproduces the stock
`yamnet` embedding to 1.06e-3 (float16 quantisation), so this is a
pipeline/regime difference, not a representation one. **This is why the
fine-tune comparator is `trunk_frozen`, not `cv-baseline`.**

### trunk_ft_1e5 vs trunk_frozen  (the effect of unfreezing)

`tools/compare_folds.py models/trunk_frozen models/trunk_ft_1e5`:

| fold | frozen | ft_1e5 | Δ | best_epoch (f/ft) |
|---|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 0.441 | 0.469 | +0.028 | 42 / 9 |
| Lily - Fit+Fast/…/53 | 0.370 | 0.432 | +0.062 | 55 / 94 |
| Lily Adam - One Hive/…/willard/…/1_11 | 0.204 | 0.318 | +0.114 | 32 / 45 |
| Lily Adam - One Hive/…/wooster/…/1_143 | 0.284 | 0.337 | +0.053 | 27 / 35 |
| Luke - Diel Drivers/2026-04-08/1_150 | 0.048 | 0.068 | +0.020 | 10 / 2 |
| Luke - Diel Drivers/2026-05-06/1_95 | 0.035 | 0.048 | +0.013 | 36 / 62 |
| Luke - Various Opp…/2025-06-23/1_23 | 0.337 | 0.400 | +0.063 | 88 / 124 |
| Luke - Various Opp…/2025-07-03/1_37 | 0.303 | 0.293 | −0.010 | 55 / 89 |
| Luke - Various Opp…/2025-08-05/31 | 0.135 | 0.185 | +0.050 | 10 / 9 |
| Luke - Various Opp…/2025-08-12/1_114 | 0.194 | 0.316 | +0.122 | 68 / 10 |
| Luke - Various Opp…/2025-08-27/48 | 0.027 | 0.020 | −0.007 | 20 / 5 |

**9 up, 2 down (both < 0.011). Mean Δ +0.046.** vs cv-baseline: 10 up, 1 down,
+0.063. val_loss is lower for ft on nearly every fold too. The two down folds
are small/quiet (299 and 307 buzz frames, thresholds resting on 6 and 22 neg
frames). The big movers include willard — the deployment temporal context
*regressed* on (`willard-regression` in IDEAS) — and Opp/2025-08-12.

Direction is not a noise artifact: most folds move the same way and several by
> 0.05, which is the bar LOOP.md sets. Size is caveated: (1) several folds
restore very early epochs (best 2, 5, 9, 10) — the small-batch noisy-val_loss /
`stopping-rule-scale` problem — though `trunk_frozen` runs the identical
stopping regime so the *paired* delta absorbs most of it; (2) `trunk_frozen`
itself sits +0.017 over `cv-baseline`, so the honest "unfreezing" number is the
+0.046 vs frozen, not +0.063 vs baseline; (3) no seed control anywhere.

### trunk_ft_5e5  (backbone LR sensitivity)

0.257, +0.041 vs frozen, 9/11 up — effectively tied with `trunk_ft_1e5`. The
gain is robust to backbone LR across 1e-5..5e-5. `best_epoch` restores are even
earlier at 5e-5 (4, 1, 6, 11 on several folds) — it overfits slightly more —
so 1e-5 is the pick.

### trunk_ft_1e5_aug  (augmentation on top)

Within-fold additive noise (`NoiseSpec(prop=0.05)`, `augment.py`) generated for
all 11 rotating folds through `yamnet_trunk` and loaded per training fold
(`load_augmented` relaxed to skip folds with no augment dir — the 311 tiny
always-train idents were not augmented; still strictly within-fold). Config
otherwise identical to `trunk_ft_1e5`.

_[Result: see `models/trunk_ft_1e5_aug/folds_sx.csv` and
`tools/compare_folds.py models/trunk_ft_1e5 models/trunk_ft_1e5_aug`. This CV
was launched last and may not have finished within the session; the aug pool
~doubles the training frames so each fold is ~2x slower. If incomplete,
`03_train/resummarize.py trunk_ft_1e5_aug` rebuilds the summary from whatever
folds completed, or re-run `cv.sh` — it resumes.]_

## Conclusion

Unfreezing YAMNet's last two blocks at a differential rate (backbone 1e-5, head
2e-4) lifts held-out sens@fpr0.005 by **+0.046 vs a matched frozen control**
(9/11 folds up), the largest structural gain in the CV log. This directly
inverts the pre-rework `yamnet-ft` verdict (−8.3pp), as `temporal-context` did
before it — the old negative was the retired corpus / fixed split, not the
mechanism. Uniform-rate fine-tuning still overfits (lite), so the differential
rate matters; `trunk_ft_5e5` probes whether a faster backbone rate helps or
tips over. Worth following: a proper LR/patience sweep on this structure (the
early-epoch restores say the stopping rule is not tuned for batch 1024), then
augmentation on top, then unfreezing more/fewer blocks.

Trust: **caveated** — direction clean and strong, size softened by the
early-stop restores and the frozen-vs-baseline pipeline offset.
