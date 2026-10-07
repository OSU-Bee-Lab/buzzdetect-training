# HANDOFF: ps-depth6 (batch 24, experiment 2 of 4)

Batch 24 status: 1 of 4 logged (ps-depth8-repeat). This one (ps-depth6) is running; ps-depth4 (experiment 3) is queued behind it in its own worktree (see ../ps-depth4/HANDOFF.md). Experiment 4 is still open: pick it when these finish. If depth6 beats depth8 by > ~0.027, run its repeat (`ps-depth6-repeat`). Otherwise pick from IDEAS.md's distillation "Untried" list.

- Job: launch_job pid **3210686**, log `.local/worktrees/ps-depth6/train.log`. Adopt it with `tools/watch_job.sh --adopt 3210686`, then watch it with one `tools/watch_job.sh` Monitor.
- **If it's still running, report progress (folds done x ~24 min) and stop.**
- Started 2026-10-06 22:52. Fold 1 trained in ~23 min, so ~8 x 24 min ≈ 3.2 h, ending ~02:05.
- **When it finishes (exit 0):** `python tools/results.py v4-ft-ps ps-depth6`, plus `tools/results.py ps-depth8-repeat ps-depth6` (ps-depth8 draws 0.515/0.509). Fill notes.md Results/Conclusion (depth ladder 12 -> 10 -> 8 -> 6: 0.452 -> 0.480 -> 0.512 -> ?), delete this HANDOFF.md, commit, then `tools/finish_experiment.sh ps-depth6 --model .local/worktrees/ps-depth6/models/ps-depth6 --baseline-model v4-ft-ps --trust clean ...` (LOOP.md step 5).
- **If it died:** read the tail of train.log. An OOM in training means lowering `TRUNK_BATCH` and raising `TRUNK_ACCUM` (keep the product 1024). An OOM in scoring/surprisal means lowering `SCORE_CHUNK`. Relaunch from this worktree; finished folds are skipped:
  `SCORE_CHUNK=256 TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=512 TRUNK_ACCUM=2 tools/launch_job.sh train.log -- 03_train/main.py --name ps-depth6 --set medium --embedder yamnet_trunk_pitchshift_depth6 --translation general --epochs 30 -y`
  Note that ps-depth4's queued job waits on pid 3210686 and starts the moment it exits. Don't launch depth6 again while depth4 holds the GPU: wait for depth4, or kill depth4 first (it resumes).
