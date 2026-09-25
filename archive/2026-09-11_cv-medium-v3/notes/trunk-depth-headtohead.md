# trunk-depth-headtohead

## Hypothesis

`trunk-ft-depth12` (2026-09-22) found unfreezing layers 12-14 gave
+0.091 ± 0.016 over its own frozen control (0.321 -> 0.412), numerically
above `trunk-ft-v3`'s own 13-14 result (+0.077, 0.321 -> 0.398) run two days
earlier. Its own notes.md flagged this as **not a resolved ranking**: no
paired per-fold SD exists against `trunk-ft-v3`, since that branch's
`models/trunk3-*` were never committed, and the two runs are two annotation
snapshots apart (2026-09-20 vs 2026-09-22).

This reruns `yamnet_trunk` (cut at layer12, tune 13-14) fresh, same
annotation snapshot as `trunk-ft-depth12`'s run (today), identical config
(`TRUNK_LR_BACKBONE`, batch, epochs), so the two depths are paired: same
frame timestamps, same day, same code. This directly answers whether cutting
one block earlier (12-14) beats the original (13-14), or whether the two
are within noise of each other on matched data -- the old E2 depth-ladder
result (12-14 worse) already failed to reproduce once; this either confirms
"depth doesn't matter much here" or actually separates the two for the
first time under this era's rules.

Prediction: given depth12's frozen control landed identically to trunk-ft-v3's
(0.321, an exact match despite different data snapshots -- the frozen cut
point costs nothing on its own), and one extra unfrozen block is a small
capacity change, I expect the fine-tuned headlines to land within this
era's ~0.012-0.016 noise floor of each other. A clean, SD-backed separation
either way would be new information.

## Changes

None -- no new code. Extracts `embedders/yamnet_trunk` (existing, cut at
layer12, tune 13-14) fresh on `medium`'s current snapshot, trains frozen +
fine-tuned arms at `trunk-ft-depth12`'s exact config.

```bash
tools/launch_job.sh extract.log -- 02_set/main.py --set medium --embedder yamnet_trunk --workers 1
TRUNK_LR_BACKBONE=0 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train_frozen.log -- 03_train/main.py --name trunk1314-frozen --set medium --embedder yamnet_trunk --translation general --epochs 60 -y
TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train_ft.log -- 03_train/main.py --name trunk1314-ft-1e5 --set medium --embedder yamnet_trunk --translation general --epochs 60 -y
```

Comparator (already trained today, read directly, not retrained):
- `trunkd12-frozen` / `trunkd12-ft-1e5`
  (`.local/worktrees/trunk-ft-depth12/models/trunkd12-*`) -- same day, same
  config, one block deeper cut.

*Falsifier:* if `trunk1314-ft-1e5` vs `trunkd12-ft-1e5` is inside the era's
~0.012-0.016 headline-delta noise floor, "depth12 also works, at least as
well" (depth12's own conclusion) is confirmed as "matched, not ranked" with
a real paired SD this time, not just two separate anchors. A clean win
either direction resolves the open question for the first time.

## Changes (addendum)

The `yamnet_trunk`-family `03_train/train.py` machinery (`TRUNK_FP16`
lowmem path, `TRUNK_BATCH`, the `build_head()` hook, batched fold scoring)
was never merged to main -- each trunk experiment has ported it by hand into
its own worktree (`trunk-ft-v3`'s docstring says so explicitly). This
worktree was created fresh from main and initially lacked it entirely, so
the first `trunk1314-frozen` launch silently ignored `TRUNK_LR_BACKBONE`/
`TRUNK_FP16`/`TRUNK_BATCH`, fell through to the plain `_to_tf` path
(batch=65568, float32, 12288-d), and OOMed the GPU on the very first fold
(`Dst tensor is not initialized`, 3.44 GiB against a 2.4 GB pool). Ported the
same hunks `trunk-ft-depth12` carries (verbatim) before relaunching.

## Results

Own matched control (`trunk1314-frozen` -> `trunk1314-ft-1e5`):

| fold | frozen | ft-1e5 | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.442 | 0.502 | +0.060 | 0.035 | 32 |
| 53 | 0.419 | 0.527 | +0.108 | 0.048 | 28 |
| 1_11 | 0.335 | 0.539 | +0.204 | 0.042 | 26 |
| 1_143 | 0.441 | 0.532 | +0.091 | 0.035 | 22 |
| 1_150 | 0.213 | 0.250 | +0.037 | 0.042 | 21 |
| 1_95 | 0.045 | 0.101 | +0.056 | 0.023 | 46 |
| 1_37 | 0.391 | 0.398 | +0.007 | 0.040 | 14 |
| 1_114 | 0.242 | 0.283 | +0.041 | 0.033 | 28 |

headline (exclquiet): 0.316 -> 0.392 (+0.076 ± 0.013), incl. quiet: 0.258 -> 0.323, 8/8 folds up.

Head-to-head vs `trunk-ft-depth12`'s `trunkd12-ft-1e5` (same day, same
config, one block deeper cut, paired by frame timestamp):

| fold | depth12 (baseline) | depth13-14 (this) | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.516 | 0.502 | -0.014 | 0.022 | 32 |
| 53 | 0.567 | 0.527 | -0.040 | 0.040 | 28 |
| 1_11 | 0.503 | 0.539 | +0.036 | 0.026 | 26 |
| 1_143 | 0.542 | 0.532 | -0.010 | 0.023 | 22 |
| 1_150 | 0.315 | 0.250 | -0.065 | 0.043 | 21 |
| 1_95 | 0.149 | 0.101 | -0.048 | 0.019 | 46 |
| 1_37 | 0.464 | 0.398 | -0.066 | 0.036 | 14 |
| 1_114 | 0.242 | 0.283 | +0.041 | 0.029 | 28 |

headline: depth12 0.412 -> depth13-14 0.392 (-0.020 ± 0.011, ~1.8 sigma),
6/8 folds down. Tier deltas (depth12 vs depth13-14): untagged -0.018,
background -0.035, loud -0.001, quiet -0.005 -- no tier moves beyond noise.

Interpretation: the fine-tuning gain itself reproduces cleanly a third time
(+0.076 ± 0.013 here, vs +0.091 ± 0.016 for depth12 and +0.077 for
`trunk-ft-v3`, all from independent frozen controls landing within
0.316-0.321 of each other). The depth question this experiment was built to
resolve does not separate: -0.020 ± 0.011 is inside this era's established
~0.012-0.016 headline noise floor, so this is "matched, not ranked" --
now with a real paired SD, not two independent anchors. No fold moves past
~2 sigma either way (1_150 and 1_37 are the largest at ~1.5-1.8 sigma,
both favoring depth12, both individually unsure). 1_95 (jet) and 1_114
(trill) both stay within noise of each other across the two depths --
neither cut is differentially better on the era's hard folds.

## Conclusion

`trunk1314-ft-1e5` (cut at layer12, tune 13-14) reruns fine-tuning fresh
against its own frozen control and confirms the lever again: +0.076 ± 0.013,
8/8 folds up, untagged (+0.081) and background (+0.086) both move with the
headline -- a real detection gain, not a background-tier artifact. The
paired head-to-head this experiment was designed for -- depth12 (cut at
layer11, tune 12-14) vs depth13-14, same day, same config -- lands within
noise (-0.020 ± 0.011, ~1.8 sigma, 6/8 folds favoring depth12 but none
individually significant). Depth of the fine-tuned cut (one block earlier
or not) does not resolve either way under this era's rules; treat the two
as equivalent defaults and prefer whichever is already in hand for future
work. Logged against `trunk1314-frozen` (this run's own matched control) so
the fine-tuning-reproduces claim carries a real paired SD; the depth12
comparison is the open question this run answers and belongs in the
conclusion text.
