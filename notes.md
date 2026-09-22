# trunk-ft-depth12

## Hypothesis

New proposal, not from IDEAS.md's queue. `trunk-ft-v3` reran unfreezing
YAMNet's last two separable-conv blocks (layers 13-14) under the current era
and found a robust +0.077, matching E2's direction (+0.046) at a larger
size. `trunk-ft-v3`'s own hypothesis section cited an unreproduced old E2
depth-ladder number: unfreezing one layer deeper (12-14) was *worse* than
13-14 there (frozen 0.216 -> 13-14 0.262 -> 12-14 0.229). That number is
pre-cutover (early stopping, old data, old metric) and per LOOP.md's standing
principle ("rerun rather than defer" -- `temporal-context` inverted between
E1 and E2), it is a lead, not an answer.

`embedders/yamnet_trunk_depth12` moves the frozen/trainable cut one block
earlier: caches layer11's activations (same (6,4,512) shape as layer12's --
both are stride-1 512-filter blocks, so this costs nothing in cache size or
frame grid) and lifts layers 12-14 + GlobalAveragePooling into the trainable
tail, instead of 13-14. Everything else -- framing, BatchNorm frozen, the
per-variable backbone LR, epoch budget -- is unchanged from `yamnet_trunk`.

Prediction: if 13-14 was already the sweet spot (a small, mostly-frozen tail
that specialises without destabilising), one layer deeper adds parameters and
moving-target depth without adding capacity where it's needed, and should be
flat-to-negative like the old E2 number. If the old E2 result was itself an
early-stopping/data artifact (plausible, since **every** E2 number has failed
to directly transfer this era), it may pay like 13-14 did.

Comparator: `trunk3-frozen` is `trunk-ft-v3`'s matched control for the
`yamnet_trunk` pipeline (batch 1024, head LR 2e-4, fp16 cache, 60 epochs) --
re-trains a `yamnet_trunk_depth12`-frozen (`lr_backbone=0`) control fresh
under the identical config, since a different cut point could in principle
change the frozen-probe number too (unlikely, since layer11's BatchNorm
stats are still AudioSet's and frozen, but untested). Primary comparator for
the fine-tuned run is `trunk3-ft-1e5` (0.398, layers 13-14 at lr_backbone=1e-5),
the confirmed lead depth.

## Changes

New embedder `embedders/yamnet_trunk_depth12/embedder.py`: subclasses
`EmbedderYamnetTrunk`, overrides `initialize()` to cut at
`layer11_pointwise_conv_relu` instead of `layer12_...`, and `build_head()` to
lift layers 12-14 + GAP as the trainable tail (mirrors the parent's
`build_head()` exactly otherwise -- same `n_ctx` path, same frozen-BatchNorm
rule, same optimizer). No `03_train` changes.

```bash
tools/launch_job.sh extract.log -- 02_set/main.py --set medium --embedder yamnet_trunk_depth12 --workers 1
TRUNK_LR_BACKBONE=0 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train_frozen.log -- 03_train/main.py --name trunkd12-frozen --set medium --embedder yamnet_trunk_depth12 --translation general --epochs 60 -y
TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train_ft.log -- 03_train/main.py --name trunkd12-ft-1e5 --set medium --embedder yamnet_trunk_depth12 --translation general --epochs 60 -y
```

(env vars TRUNK_LR_BACKBONE / TRUNK_LR_HEAD / TRUNK_BATCH / TRUNK_FP16 and the
60-epoch, batch-1024 config are `yamnet_trunk`'s own -- see `trunk-ft-v3`'s
notes.md on `exp/trunk-ft-v3` for how `_train_one` reads them via
`embedder.build_head`. Ported the `03_train/train.py` hook and `chain.sh`-style
launch from that branch since it hasn't merged to main. **Learned the hard
way**: omitting `TRUNK_FP16=1`/`TRUNK_BATCH=1024` isn't a no-op default -- the
plain `_to_tf` path tries to materialise the full 12288-d array as float32
(~3.5 GB) and the default 65568 batch is effectively the whole fold in one
step; both OOM the 4 GB card immediately. `trunk-ft-v3`'s own budget table
(GPU probe on this exact card) is why 1024 was its answer, not 4096.)

## Results

GPU OOM'd twice during this run (once immediately, from omitting
`TRUNK_FP16=1`/`TRUNK_BATCH=1024` -- not optional defaults at this width, see
Changes note; once genuinely mid-run on the 4 GB card, on the frozen-control
retry AND again on the fine-tuned run's 8th fold). Also caught a real
resume-detection gap: the fine-tuned run's OOM left fold `1_37` with
`config_model.json` + `predictions.csv` written but no `summary.json`, and
the resume logic (`can_write()`, gated on `config_model.json`) reported it
"already trained; skipping" although it had silently dropped out of
`folds_sx.csv`'s mean. Caught by checking the fold list, not the headline;
flagged as friction, since this could quietly turn an 8-fold headline into a
7-fold one on any interrupted run. Fixed by deleting that one fold's
directory and rerunning it.

Also ran the full `--epochs 400` budget by mistake on the first frozen-control
attempt (the default, not `trunk-ft-v3`'s chosen 60) before catching it and
killing/relaunching with `--epochs 60` to match the comparator's budget.

| fold | frozen (`lr_backbone=0`) | ft (`lr_backbone=1e-5`, layers 12-14) | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.450 | 0.516 | +0.066 | 0.043 | 32 |
| 53 | 0.414 | 0.567 | +0.153 | 0.069 | 28 |
| 1_11 (willard) | 0.373 | 0.503 | +0.130 | 0.030 | 26 |
| 1_143 | 0.459 | 0.542 | +0.083 | 0.033 | 22 |
| 1_150 | 0.222 | 0.315 | +0.093 | 0.051 | 21 |
| 1_95 | 0.042 | 0.149 | +0.107 | 0.033 | 46 |
| 1_37 | 0.407 | 0.464 | +0.057 | 0.045 | 14 |
| 1_114 | 0.202 | 0.242 | +0.040 | 0.033 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): `trunkd12-frozen` 0.321 ->
  `trunkd12-ft-1e5` 0.412 (**+0.091 ± 0.016**), 8/8 folds up
- inclusive (sensitivity), same thresholds: 0.262 -> 0.342 (+0.080)
- folds that missed fpr 0.005: none

| tier | frozen | ft | delta | frames |
|---|---|---|---|---|
| loud | 0.787 | 0.787 | +0.000 | 122 |
| untagged | 0.341 | 0.433 | +0.092 | 2418 |
| background | 0.320 | 0.441 | +0.121 | 1874 |
| quiet | 0.071 | 0.103 | +0.032 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

**Against `trunk-ft-v3`'s own 13-14 result** (headline only -- its
`models/trunk3-*` were never committed to `exp/trunk-ft-v3`, so no paired
per-fold SD is possible, only the numbers in its notes.md/log.jsonl):
`trunk3-frozen` 0.321 -> `trunk3-ft-1e5` 0.398 (+0.077). This run's frozen
control landed at the *identical* 0.321 (reassuring -- the one-block-earlier
cut costs nothing on its own, as predicted), and depth12's fine-tuned arm
(0.412) is **numerically above** 13-14's 0.398, +0.014 apart -- inside this
era's ~0.012-0.016 headline-delta noise floor, so "deeper is a little
better" is not resolved by one run each; call it matched, not ranked.

## Conclusion

**Real gain, and the old E2 depth-ladder verdict (12-14 worse than 13-14)
does not reproduce under this era's rules.** +0.091 ± 0.016 (>5σ) over the
frozen control, 8/8 folds up -- larger and cleaner than `trunk-ft-v3`'s own
13-14 result (+0.077) on its own comparator, though the two fine-tuned
headlines (0.412 vs 0.398) sit inside the era's noise floor of each other,
so this is "depth12 also works, at least as well," not "depth12 beats
depth13-14." Both named hard folds move up: `1_95` (jet) +0.107 ± 0.033
(~3.2σ) and `1_114` (trill) +0.040 ± 0.033 (~1.2σ, weak); `1_150` +0.093 ±
0.051 (~1.8σ, moderate). `untagged` (+0.092, 2418 frames) and `background`
(+0.121) both move with the headline -- a genuine detection gain, not a
single-tier artifact; `loud` and `faint` are flat (small samples).

Confirms LOOP.md's "rerun rather than defer" principle again: the E2 number
this hypothesis was testing against (12-14 worse than 13-14, under early
stopping and old data) does not hold under the fixed-epoch regime. Whether
a fold-matched head-to-head (same annotation snapshot, same code, both
depths in one run) would separate 12-14 from 13-14 is the open follow-up --
not run here since the headline delta is inside noise either way.
