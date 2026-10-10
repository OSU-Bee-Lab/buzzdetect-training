# HANDOFF: ps-d8-foldbal

Batch 29 (4 experiments; **2 logged as of 19:35 on 2026-10-10: `ps-depth8-redraw` (0.497) and `ps-d8-taps` (0.481)**; `ps-d8-foldbal` and `fast32-d8-repeat` remain). pid 2744321 has exited; only the chain (2780967) is alive: taps exit 0 at 19:15, foldbal started then (ends ~21:25), distill follows (~5 h). In running order: `ps-depth8-redraw`, `ps-d8-taps`, `ps-d8-foldbal` (training arm), `fast32-d8-repeat` (distillation arm). Each has its own HANDOFF.md; an experiment is done when its slug is in `03_train/log.jsonl` (or, for the distill one, its students carry `"exp"` in `05_distill/log.jsonl`).

Jobs (both started by `tools/launch_job.sh`; adopt each with `tools/watch_job.sh --adopt <pid>`, then arm the one `tools/watch_job.sh` Monitor):
- pid 2744321, log `.local/worktrees/ps-depth8-redraw/train.log`: the control CV `ps-depth8-r3` (8 folds, ~16 min each, started 14:57).
- pid 2780967, log `.local/worktrees/ps-d8-taps/chain.log`: one chain that waits for 2744321 to exit, then runs the `ps-d8-taps` CV, the `ps-d8-foldbal` CV and the `fast32-d8-repeat` distillation, in that order, whatever each exits with. It prints `[chain] <step> exit N` after each step. Expected: taps ends ~19:20, foldbal ~21:30, distill ~5 h after that.

If the job this handoff needs is still running: log whatever experiments have already finished (their own handoffs), report progress, and park again (`loop_signal.sh park <minutes>`), keeping the handoffs of the unfinished ones. Delete a HANDOFF.md (git rm, commit) before `finish_experiment.sh` on its slug.

All four use env `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024` and args `--set medium --embedder yamnet_trunk_pitchshift_depth8 --translation general --epochs 30 -y` where a training command is given below; reruns resume at the first unfinished fold. One GPU job at a time.

## This experiment
Job: the chain, pid 2780967, step 2 (`[chain] foldbal exit N` in `.local/worktrees/ps-d8-taps/chain.log`). Model `models/ps-d8-foldbal` in this worktree. Smoke-tested on lite. The log prints `[foldbal] p=0.5: N folds, frame weight min/median/max` per rotation: quote it in notes.md, and replace the Hypothesis section's rough "1 to 120 units" fold-size range with real frame counts if you have them.

When it finishes: check `buzz logit SD` per fold, then `$(bash tools/python_path.sh) tools/results.py ps-depth8-r3 ps-d8-foldbal` (also quote the delta against `ps-depth8-repeat`). Fill notes.md per its Hypothesis section, then `tools/finish_experiment.sh ps-d8-foldbal --model .local/worktrees/ps-d8-foldbal/models/ps-d8-foldbal --baseline-model ps-depth8-r3 ...` from main. If `ps-depth8-r3` never completed, baseline `ps-depth8-repeat`, trust `caveated`.

If it died (or the loop killed the chain): from this worktree, `TRUNK_FOLDBAL=0.5 TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train.log -- 03_train/main.py --name ps-d8-foldbal --set medium --embedder yamnet_trunk_pitchshift_depth8 --translation general --epochs 30 -y`.
