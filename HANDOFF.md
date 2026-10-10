# HANDOFF: ps-d8-taps

Batch 29 (4 experiments, 0 logged when this was written at 15:12 on 2026-10-10). In running order: `ps-depth8-redraw`, `ps-d8-taps`, `ps-d8-foldbal` (training arm), `fast32-d8-repeat` (distillation arm). Each has its own HANDOFF.md; an experiment is done when its slug is in `03_train/log.jsonl` (or, for the distill one, its students carry `"exp"` in `05_distill/log.jsonl`).

Jobs (both started by `tools/launch_job.sh`; adopt each with `tools/watch_job.sh --adopt <pid>`, then arm the one `tools/watch_job.sh` Monitor):
- pid 2744321, log `.local/worktrees/ps-depth8-redraw/train.log`: the control CV `ps-depth8-r3` (8 folds, ~16 min each, started 14:57).
- pid 2780967, log `.local/worktrees/ps-d8-taps/chain.log`: one chain that waits for 2744321 to exit, then runs the `ps-d8-taps` CV, the `ps-d8-foldbal` CV and the `fast32-d8-repeat` distillation, in that order, whatever each exits with. It prints `[chain] <step> exit N` after each step. Expected: taps ends ~19:20, foldbal ~21:30, distill ~5 h after that.

If the job this handoff needs is still running: log whatever experiments have already finished (their own handoffs), report progress, and park again (`loop_signal.sh park <minutes>`), keeping the handoffs of the unfinished ones. Delete a HANDOFF.md (git rm, commit) before `finish_experiment.sh` on its slug.

All four use env `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024` and args `--set medium --embedder yamnet_trunk_pitchshift_depth8 --translation general --epochs 30 -y` where a training command is given below; reruns resume at the first unfinished fold. One GPU job at a time.

## This experiment
Job: the chain, pid 2780967, step 1 (`[chain] taps exit N` in chain.log). Model `models/ps-d8-taps` in this worktree. Smoke-tested on lite (`[taps] readout width 4096`).

When it finishes: check `buzz logit SD` per fold in chain.log, then `$(bash tools/python_path.sh) tools/results.py ps-depth8-r3 ps-d8-taps` (control = this batch's redraw; also quote the delta against `ps-depth8-repeat`). Fill notes.md per its Hypothesis section (falsifiers are written there), then `tools/finish_experiment.sh ps-d8-taps --model .local/worktrees/ps-d8-taps/models/ps-d8-taps --baseline-model ps-depth8-r3 ...` from main. If the control `ps-depth8-r3` never completed, use `ps-depth8-repeat` as the baseline and mark trust `caveated` (stale control). A gain over ~0.03 gets one repeat run before it counts.

If it died (or the loop killed the chain): from this worktree, `TRUNK_TAPS=layer10_pointwise_conv_relu,layer12_pointwise_conv_relu TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train.log -- 03_train/main.py --name ps-d8-taps --set medium --embedder yamnet_trunk_pitchshift_depth8 --translation general --epochs 30 -y`. On a GPU OOM in the train step, add `TRUNK_BATCH=512 TRUNK_ACCUM=2` (same update) and clear nothing: finished folds are kept.
