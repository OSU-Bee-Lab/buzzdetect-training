# HANDOFF: ps-depth8-redraw

Batch 29 (4 experiments, 0 logged when this was written at 15:12 on 2026-10-10). In running order: `ps-depth8-redraw`, `ps-d8-taps`, `ps-d8-foldbal` (training arm), `fast32-d8-repeat` (distillation arm). Each has its own HANDOFF.md; an experiment is done when its slug is in `03_train/log.jsonl` (or, for the distill one, its students carry `"exp"` in `05_distill/log.jsonl`).

Jobs (both started by `tools/launch_job.sh`; adopt each with `tools/watch_job.sh --adopt <pid>`, then arm the one `tools/watch_job.sh` Monitor):
- pid 2744321, log `.local/worktrees/ps-depth8-redraw/train.log`: the control CV `ps-depth8-r3` (8 folds, ~16 min each, started 14:57).
- pid 2780967, log `.local/worktrees/ps-d8-taps/chain.log`: one chain that waits for 2744321 to exit, then runs the `ps-d8-taps` CV, the `ps-d8-foldbal` CV and the `fast32-d8-repeat` distillation, in that order, whatever each exits with. It prints `[chain] <step> exit N` after each step. Expected: taps ends ~19:20, foldbal ~21:30, distill ~5 h after that.

If the job this handoff needs is still running: log whatever experiments have already finished (their own handoffs), report progress, and park again (`loop_signal.sh park <minutes>`), keeping the handoffs of the unfinished ones. Delete a HANDOFF.md (git rm, commit) before `finish_experiment.sh` on its slug.

All four use env `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024` and args `--set medium --embedder yamnet_trunk_pitchshift_depth8 --translation general --epochs 30 -y` where a training command is given below; reruns resume at the first unfinished fold. One GPU job at a time.

## This experiment
Job: pid 2744321. Model `models/ps-depth8-r3` in this worktree.

When it finishes (`[launch_job] exit 0`, `folds_sx.csv` has a `total` row): check every fold line's `buzz logit SD` in train.log (0.65-1.2 is trained), then `$(bash tools/python_path.sh) tools/results.py ps-depth8-repeat ps-depth8-r3` and also against `ps-depth8` and `pseudo-label_teacher`. Read `docs/judging-results.md`. Fill notes.md Results/Conclusion: does main still reproduce ~0.51 (then the 0.466 teacher draw belongs to the PSEUDO_SET path or to noise), or has it drifted to ~0.47 (then say so loudly: the era's best control is stale, and bisecting 03_train since 2026-10-01 is the next experiment). Then `tools/finish_experiment.sh ps-depth8-redraw --model .local/worktrees/ps-depth8-redraw/models/ps-depth8-r3 --baseline-model ps-depth8-repeat ...` from main (LOOP.md step 5). No IDEAS.md entry to delete.

If it died: from this worktree, `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train.log -- 03_train/main.py --name ps-depth8-r3 --set medium --embedder yamnet_trunk_pitchshift_depth8 --translation general --epochs 30 -y`. If the chain (2780967) is alive it starts the taps CV the moment 2744321 exits, so the GPU will be taken: wait for the chain to end, or if the loop killed it, relaunch this first.
