# HANDOFF — exp/trunk-pitchshift-contrast

**Halted for a loop restart Luke requested (2026-09-22), not a technical
blocker.** Nothing is broken. No job is left running for this experiment
specifically (see below for why) — it's next in the GPU queue behind
`exp/trunk-depth-headtohead`'s `trunk1314-ft-1e5`.

## What's done

- New embedder `embedders/yamnet_trunk_pitchshift_contrast` (shared tree,
  committed to main at `0b82ff1`): the fine-tuned trunk tail from
  `yamnet_trunk_pitchshift`, but the two 1024-d codes are joined with the
  explicit signed contrast (`[plain, plain - shifted]`) instead of a plain
  concat — see notes.md Hypothesis for the two results (`pitchshift-contrast`
  vs `trunk-ft-pitchshift`) this is trying to reconcile.
- Lite smoke test: clean (graph builds, trains, no shape errors).
- `medium` extraction: **done**, exit 0 (`extract.log`).

## Nothing running — do this next

GPU is currently held by `exp/trunk-depth-headtohead`'s `trunk1314-ft-1e5`
(its own HANDOFF.md has the pid to watch). Once that's done (or if you
check and the GPU is actually free), launch this experiment's two training
arms, same config every other trunk-family experiment this era used:

```bash
cd .local/worktrees/trunk-pitchshift-contrast
TRUNK_LR_BACKBONE=0 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train_frozen.log -- \
  03_train/main.py --name trunkpscontrast-frozen --set medium \
  --embedder yamnet_trunk_pitchshift_contrast --translation general --epochs 60 -y
TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train_ft.log -- \
  03_train/main.py --name trunkpscontrast-ft-1e5 --set medium \
  --embedder yamnet_trunk_pitchshift_contrast --translation general --epochs 60 -y
```

Run them one at a time (only one GPU job at once). Watch each with
`tools/watch_job.sh <pid> <log>` in a Monitor, re-armed every 30 min.

**Important:** this worktree's `03_train/train.py` was created fresh from
main and does **not** have the `TRUNK_FP16`/`build_head()`/batched-scoring
machinery main never got merged — the same gap that caused a GPU OOM in
`exp/trunk-depth-headtohead` this batch (see that experiment's notes.md and
the friction report filed 2026-09-22). Before launching, port the same hunks
from `.local/worktrees/trunk-ft-depth12/03_train/train.py` (or
`trunk-depth-headtohead`'s, now fixed) into this worktree's `03_train/train.py`:
`_rss_gb`, `_to_tf`/`_to_tf_lowmem`/`_Fp16Batches`, `TRUNK_BATCH` env var,
`free=True` on the train call, batched `np.concatenate` scoring, and the
`build_head()` hook in the optimizer/model-construction block. Diff the two
files to find the exact hunks if unsure.

## When both finish

1. Compare `trunkpscontrast-ft-1e5` against `trunkpscontrast-frozen` (own
   matched control) and against `trunk-ft-pitchshift`'s `trunkps-ft-1e5`
   (`.local/worktrees/trunk-ft-pitchshift/models/trunkps-ft-1e5` — same tail
   depth and pitch-shift mechanism, concat instead of contrast):
   ```bash
   conda run -n buzzdetect-train python tools/results.py \
     models/trunkpscontrast-frozen models/trunkpscontrast-ft-1e5
   conda run -n buzzdetect-train python tools/results.py \
     /home/luke/projects/buzzdetect-training/.local/worktrees/trunk-ft-pitchshift/models/trunkps-ft-1e5 \
     models/trunkpscontrast-ft-1e5
   ```
2. **`1_114` is the named falsifier** (see notes.md): does the contrast
   design recover `1_114`'s clean win (like frozen `pitchshift-contrast`,
   +0.153) while keeping fine-tuning's gain — unlike `trunk-ft-pitchshift`'s
   concat, which lost `1_114` (-0.038)?
3. Write Results + Conclusion in notes.md, then
   `tools/finish_experiment.sh trunk-pitchshift-contrast ...` from main.

## If a training job dies

Re-running the same command resumes (finished folds are skipped via
`can_write()`). If GPU OOM persists after porting the lowmem fix, fall back
to `--cpu` on `launch_job.sh`.
