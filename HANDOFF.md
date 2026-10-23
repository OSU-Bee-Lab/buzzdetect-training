# HANDOFF: pitch-aux-d8 (batch 42)

Batch 42's four experiments: pitch-aux-d8, noquiet-d8, bn-site-d8, bn-shrunk-d8. **None logged yet** at park time (2026-10-23 15:40).

Jobs (one launch_job each; adopt both with `tools/watch_job.sh --adopt <pid>`, then ONE `tools/watch_job.sh` Monitor):
- pid 2307151, log `.local/worktrees/pitch-aux-d8/chain_b42.log`: `chain_b42.sh` = pitch-aux-d8 CV, then noquiet-d8 CV. ~16.5 min/fold, 8 folds each.
- pid 2314980, log `.local/worktrees/bn-site-d8/chain_b42b.log`: waits for 2307151, then bntrain-d8 CV and the offline bn-site-d8 / bn-shrunk-d8 rescoring (see bn-site-d8/HANDOFF.md).

**If it's still running, report progress and stop** (park again if > 1.5 h remain).

This experiment's part: model `models/pitch-aux-d8` in this worktree (translation `general_pitch`).
When its CV is done (folds_sx.csv exists):
1. Check each fold's `buzz logit SD` in `chain_b42.log` (0.65-1.2 is healthy).
2. `$(bash tools/python_path.sh) tools/results.py ps-depth8-r3 .local/worktrees/pitch-aux-d8/models/pitch-aux-d8` (control: ps-depth8-r3, 0.497).
3. LOOP.md steps 4-5: fill notes.md Results/Conclusion, delete IDEAS.md #38 in main, `git rm HANDOFF.md`, then `tools/finish_experiment.sh pitch-aux-d8 --model .local/worktrees/pitch-aux-d8/models/pitch-aux-d8 --baseline-model ps-depth8-r3 ...`.

If it died: read chain_b42.log's tail, fix, then relaunch from this worktree (resumes finished folds and skips done smokes):
`cd .local/worktrees/pitch-aux-d8 && /home/luke/projects/buzzdetect-training/tools/launch_job.sh chain_b42.log -- bash chain_b42.sh`
and relaunch chain_b42b after it: `cd ../bn-site-d8 && .../tools/launch_job.sh chain_b42b.log -- bash chain_b42b.sh <new chain_b42 pid>`.
