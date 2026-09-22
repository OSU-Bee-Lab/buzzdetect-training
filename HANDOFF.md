# HANDOFF — exp/trunk-depth-headtohead

**Halted for a loop restart Luke requested (2026-09-22), not a technical
blocker.** Nothing is broken; the previous session just had to end mid-run.
The job below is left running (a `halt` signal, per `tools/loop_signal.sh`,
exits the driver before it kills any job).

## What's done

- `trunk1314-frozen` (backbone frozen, matched control): finished cleanly.
  Headline sens@fpr0.005 (exclquiet) **0.316** (incl. quiet: 0.258) — matches
  `trunk-ft-depth12`'s own frozen control (0.321) closely, as expected (same
  cut point, one block earlier than 13-14... no, same cut as depth12's
  comparator; see notes.md Hypothesis for exactly which depth this reruns).

## What's running

`trunk1314-ft-1e5` (backbone LR 1e-5, the fine-tuned arm — the actual
comparison this experiment exists to make): pid **1261468**, log
`.local/worktrees/trunk-depth-headtohead/train_ft.log`. Watch with
`tools/watch_job.sh 1261468 .../train_ft.log` in a Monitor, re-armed every
30 min (CLAUDE.md "Running long jobs"). Was on fold 3-4/8 as of the last
check (2026-09-22 17:32), ~60 epochs/fold, ~25-30 min/fold on GPU — expect
it to finish in roughly 1.5-2 h from when you pick this up.

## When it finishes

1. Run the paired comparison against `trunkd12-ft-1e5`
   (`.local/worktrees/trunk-ft-depth12/models/trunkd12-ft-1e5`) — the fair,
   same-day, same-config head-to-head this experiment is FOR:
   ```bash
   cd .local/worktrees/trunk-depth-headtohead
   conda run -n buzzdetect-train python tools/results.py \
     /home/luke/projects/buzzdetect-training/.local/worktrees/trunk-ft-depth12/models/trunkd12-ft-1e5 \
     models/trunk1314-ft-1e5
   ```
2. Also read it against this run's own frozen control
   (`models/trunk1314-frozen`) to confirm the fine-tuning gain reproduces:
   ```bash
   conda run -n buzzdetect-train python tools/results.py \
     models/trunk1314-frozen models/trunk1314-ft-1e5
   ```
3. Write the Results section in `notes.md` (template already has the
   Hypothesis/Changes sections; append Results + Conclusion under LOOP.md
   step 5's format). The falsifier is stated in notes.md: is the
   `trunk1314-ft-1e5` vs `trunkd12-ft-1e5` delta inside the era's
   ~0.012-0.016 headline-delta noise floor, or a real separation?
4. From main: `tools/finish_experiment.sh trunk-depth-headtohead ...` per
   LOOP.md step 5. Baseline model for the log entry should be
   `trunk1314-frozen` (this run's own matched control); mention the
   `trunkd12-ft-1e5` comparison in the conclusion text.

## If it died

Read `train_ft.log`'s tail. Re-running the exact same command resumes —
`trunk1314-frozen` is untouched (already complete, `can_write()` skips it),
and any finished folds of `trunk1314-ft-1e5` are skipped too:

```bash
cd .local/worktrees/trunk-depth-headtohead
TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train_ft.log -- \
  03_train/main.py --name trunk1314-ft-1e5 --set medium --embedder yamnet_trunk \
  --translation general --epochs 60 -y
```

If it OOMs the GPU again (this worktree's `03_train/train.py` had to be
hand-patched with the `TRUNK_FP16`/`build_head` machinery — see notes.md
Changes addendum and the `friction` report filed this batch), the fix is
already in place; a fresh OOM would be a new issue, not this one recurring.
If GPU OOM persists despite the fix, fall back to `--cpu` on `launch_job.sh`
(this era's precedent: `trunk-ft-depth12` needed it after repeated OOMs).
