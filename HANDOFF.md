# HANDOFF — exp/ps-depth8 (batch 21, experiment 4 of 4)

Batch 21 state: 3 of 4 experiments logged (ps-ema, ps-untied, ps-depth10-repeat). Only ps-depth8 remains.

- Job: `launch_job.sh` pid 1285788, log `.local/worktrees/ps-depth8/chain.log` (extract then train chain, wrapper script in the job tmp dir; extraction finished, training started 18:56, ~16 min/fold, ETA ~21:05).
- Resume: `tools/watch_job.sh --adopt 1285788`, then one `tools/watch_job.sh` Monitor.
- **If it's still running, report progress and stop (park again).**
- When it finishes (`[launch_job] exit 0`), from the worktree:
  `python tools/results.py /home/luke/projects/buzzdetect-training/models/v4-ft-ps models/ps-depth8` (bare name fails: worktree stub of v4-ft-ps has no predictions.csv; use main's absolute path). Also compare to ps-depth10-repeat (`/home/luke/projects/buzzdetect-training/.local/worktrees/ps-depth10-repeat/models/ps-depth10-repeat`).
  Then fill notes.md Results/Conclusion (hypothesis is written; compare depth 8 vs depth 10's 0.473/0.487, say which tiers moved, per-fold vs SD) and from main run `tools/finish_experiment.sh ps-depth8 --summary ... --model .local/worktrees/ps-depth8/models/ps-depth8 --baseline-model /home/luke/projects/buzzdetect-training/models/v4-ft-ps --hypothesis ... --trust clean --conclusion ...`. Then `loop_signal.sh done "ps-ema, ps-untied, ps-depth10-repeat, ps-depth8"`.
- If it died: rerun the same chain (resumes extraction and finished folds):
  `cd /home/luke/projects/buzzdetect-training/.local/worktrees/ps-depth8 && BUZZDETECT_CHUNK_FRAMES=48 TRUNK_FP16=1 TRUNK_LR_HEAD=2e-4 TRUNK_LR_BACKBONE=1e-5 TRUNK_BATCH=1024 tools/launch_job.sh train.log -- 03_train/main.py --name ps-depth8 --set medium --embedder yamnet_trunk_pitchshift_depth8 --translation general --epochs 30 -y --verbose`
  (if extraction is incomplete, first `tools/launch_job.sh extract.log -- 02_set/main.py --set medium --embedder yamnet_trunk_pitchshift_depth8 --workers 1 --verbose`).
