# HANDOFF: noquiet-d8 (batch 42)

Batch 42's four: pitch-aux-d8 (**logged** 2026-10-23, -0.004), noquiet-d8, bn-site-d8, bn-shrunk-d8 (three remain).

Jobs (adopt both with `tools/watch_job.sh --adopt <pid>`, then ONE `tools/watch_job.sh` Monitor):
- pid 2307151, log `.local/worktrees/pitch-aux-d8/chain_b42.log`: `chain_b42.sh`; pitch-aux-d8 CV done, now noquiet-d8 CV (fold 1 started 17:19, ~16.5 min/fold, 8 folds -> ~19:30).
- pid 2314980, log `.local/worktrees/bn-site-d8/chain_b42b.log`: waits for 2307151, then bntrain-d8 CV + soup rescoring (see bn-site-d8/HANDOFF.md).

**If it's still running, report progress and stop.**

When `models/noquiet-d8/folds_sx.csv` exists in this worktree:
1. Check buzz logit SD per fold in chain_b42.log.
2. `$(bash tools/python_path.sh) tools/results.py ps-depth8-r3 .local/worktrees/noquiet-d8/models/noquiet-d8` (control ps-depth8-r3, 0.497). Read the quiet tier beside the headline; untagged/loud are the prediction.
3. Note in Changes: dropping quiet samples also raises ins_buzz's inverse-frequency weight slightly (fewer buzz samples; `build_weights` not pinned). 76 of 1191 training samples dropped in the 1_150 rotation.
4. LOOP.md step 5: delete IDEAS.md #39 in main, `git rm HANDOFF.md`, `tools/finish_experiment.sh noquiet-d8 --model .local/worktrees/noquiet-d8/models/noquiet-d8 --baseline-model ps-depth8-r3 ...`.

If it died: read chain_b42.log's tail, fix, relaunch (resumes finished folds, skips done smokes and pitch-aux-d8's finished CV):
`cd .local/worktrees/pitch-aux-d8 && /home/luke/projects/buzzdetect-training/tools/launch_job.sh chain_b42.log -- bash chain_b42.sh`
then relaunch chain_b42b after it: `cd ../bn-site-d8 && /home/luke/projects/buzzdetect-training/tools/launch_job.sh chain_b42b.log -- bash chain_b42b.sh <new chain_b42 pid>`.
