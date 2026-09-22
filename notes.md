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
