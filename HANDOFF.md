# HANDOFF llrd-depth4
## Batch 38 state
4 experiments in this batch, 3 logged (lp-ft, pauc-l03, mixstyle-dep: all negative). Remaining: llrd-depth4 only.
- The b38 chain (pid 3875542) died at 03:25: llrd-depth4's lite smoke needs yamnet_trunk_pitchshift_depth4 embeddings, which lite lacks. mixstyle-dep had already finished and is logged.
- llrd-depth4 relaunched without a smoke at 03:27: launch_job pid 28738, log .local/worktrees/llrd-depth4/chain_llrd.log (script chain_llrd.sh in this worktree; CV output tees to train.log). ~5 h (TRUNK_STREAM, batch 256x4), ETA ~08:30.

Resume: `tools/watch_job.sh --adopt 28738`, then ONE `tools/watch_job.sh` Monitor. If still running, report progress and stop (park again if >1.5 h left).

## On finish
Comparators: ps-depth4 (0.523) as matched control, plus the three-draw depth8 mean 0.507. `$(bash tools/python_path.sh) tools/results.py ps-depth4 llrd-depth4`, check every fold line's buzz logit SD, read docs/judging-results.md, write notes.md Results/Conclusion, delete this HANDOFF.md, delete #35 (llrd-depth4) from main's IDEAS.md, then tools/finish_experiment.sh from main (LOOP.md step 5) with --commit-also IDEAS.md. Then `loop_signal.sh done "lp-ft pauc-l03 mixstyle-dep llrd-depth4"`.

## If it died
Read chain_llrd.log for the cause. Once no GPU job runs, relaunch from this worktree: `tools/launch_job.sh chain_llrd.log -- bash chain_llrd.sh` (finished folds are skipped).
