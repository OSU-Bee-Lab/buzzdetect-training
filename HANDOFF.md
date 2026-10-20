# HANDOFF bn-pooled
## Batch 39 state (updated 13:25 2026-10-20, second park)
4 experiments, 1 logged (mixstyle-xfold, a849067). Remaining: ps-d8-soup, bn-pooled (chain 2), pauc-l01 (chain 3).
- chain 1 (pid 271991) died 10:50 after mixstyle-xfold: exp/pauc-l01 lacked train_utils.pauc_aux_loss (ImportError). Fixed on exp/pauc-l01 (port from exp/pauc-l03).
- chain 2: launch_job pid 297592, log .local/worktrees/ps-d8-soup/chain2_b39.log (`ps-d8-soup/chain2_b39.sh`), started its CVs 10:51 when chain 1 died: m1 done 13:00 (2h10m), m2 running, m3 next, then offline scoring (ps-d8-soup-ens, ps-d8-soup, -w25/w50/w75; then bn-pooled, bn-pooled-soup in bn-pooled). ETA ~17:30-18:30.
- chain 3: launch_job pid 429588, log .local/worktrees/pauc-l01/chain3_b39.log (`pauc-l01/chain3_b39.sh`): waits for 297592 to exit, then execs mixstyle-xfold/chain_b39.sh, which skips mixstyle-xfold (folds_sx.csv exists) and runs pauc-l01's lite smoke + medium CV (~2.2 h). ETA ~20:30-21:00.

Resume: `tools/watch_job.sh --adopt 297592` and `--adopt 429588` (skip a dead pid), then ONE `tools/watch_job.sh` Monitor. If still running, report progress and stop (park again if >1.5 h remain), logging whichever experiments have finished.

Steps 4-5 per experiment when its models exist: `$(bash tools/python_path.sh) tools/results.py <control> <model>`, check each fold line's buzz logit SD, read docs/judging-results.md, write notes.md Results/Conclusion, delete this HANDOFF.md (commit only notes.md/HANDOFF.md: `git commit -a` in a worktree also commits the embedders/ symlink swap), delete/trim the idea in main's IDEAS.md, then tools/finish_experiment.sh from main (LOOP.md step 5). ps-depth8-r3 lives at .local/worktrees/ps-depth8-redraw/models/ps-depth8-r3 (pass that path as --baseline-model). Three-draw depth8 mean is 0.507 (ps-depth8 0.515, ps-depth8-repeat 0.509, ps-depth8-r3 0.497); the soup members add three more draws.

## This experiment
Models in .local/worktrees/bn-pooled/models/: bn-pooled (m1 + AdaBN; control is .local/worktrees/ps-d8-soup/models/ps-d8-soup-m1, the exact same weights) and bn-pooled-soup (control ps-d8-soup). Last step of chain 2; relaunch as in ps-d8-soup's HANDOFF. IDEAS #37: trim to (b)/(c) only (they stay a Luke call).
