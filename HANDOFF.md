# HANDOFF mixstyle-dep
## Batch 38 state
4 experiments in this batch, 2 logged so far (lp-ft, negative; pauc-l03, logged 2026-10-20 01:35, -0.091, negative). Remaining: mixstyle-dep (medium CV since 01:15 on chain pid 3875542, ETA ~03:25), then llrd-depth4 (~5 h, ETA ~08:30).
- lp-ft and pauc-l03: finished and logged.
- chain: launch_job pid 3875542, log .local/worktrees/pauc-l03/chain_b38.log (`pauc-l03/chain_b38.sh`). It waits for lp-ft, then for each of pauc-l03, mixstyle-dep, llrd-depth4: a 1-epoch lite smoke, then the medium CV, tee-d to that worktree's train.log. pauc/mixstyle ~2.3 h each, llrd-depth4 ~5 h (TRUNK_STREAM, batch 256x4).
Resume: `tools/watch_job.sh --adopt 3875542`, then ONE `tools/watch_job.sh` Monitor. If still running, report progress and stop (park again if >1.5 h left on the next unlogged experiment).

Comparators: matched control ps-depth8-r3 (0.497) and the three-draw depth8 mean 0.507 (ps-depth8 0.515, ps-depth8-repeat 0.509, r3 0.497) for lp-ft / pauc-l03 / mixstyle-dep; ps-depth4 (0.523) plus the depth8 mean for llrd-depth4. On finish: `$(bash tools/python_path.sh) tools/results.py <control> <slug>`, check every fold line's buzz logit SD, read docs/judging-results.md, write notes.md Results/Conclusion, delete this HANDOFF.md, delete the idea's section from main's IDEAS.md (#33 pauc-l03 -- trim to the untested lam 0.1 rung, #31 mixstyle-dep -- leave the cross-fold variant as a Luke policy call, #35 llrd-depth4), then tools/finish_experiment.sh from main (LOOP.md step 5).

## This experiment
mixstyle-dep is a step of the chain pid 3875542 (its CV output tees to .local/worktrees/mixstyle-dep/train.log, model in .local/worktrees/mixstyle-dep/models/mixstyle-dep). If the chain died, relaunch it from .local/worktrees/pauc-l03 once no GPU job runs: edit chain_b38.sh's `while kill -0 3861249` wait line out (or leave it: a dead pid passes), then `tools/launch_job.sh chain_b38.log -- bash chain_b38.sh`. Steps whose models/<slug>/folds_sx.csv exists skip the smoke and resume (finished folds are skipped).
