# HANDOFF ps-d8-soup
## Batch 39 state
4 experiments, 0 logged: mixstyle-xfold, pauc-l01 (chain 1), ps-d8-soup, bn-pooled (chain 2).
- chain 1: launch_job pid 271991, log .local/worktrees/mixstyle-xfold/chain_b39.log (`mixstyle-xfold/chain_b39.sh`): lite smoke + medium CV of mixstyle-xfold, then pauc-l01; each CV tees to its worktree's train.log. Started 08:36 2026-10-20, ~2.2 h per CV.
- chain 2: launch_job pid 297592, log .local/worktrees/ps-d8-soup/chain2_b39.log (`ps-d8-soup/chain2_b39.sh`): waits for chain 1, then three depth8 CVs with TRUNK_SAVE_WEIGHTS=1 (ps-d8-soup-m1/m2/m3, train_m*.log in ps-d8-soup), then offline scoring: ps-d8-soup-ens (logit ensemble), ps-d8-soup (uniform soup), ps-d8-soup-w25/w50/w75 (WiSE), then in bn-pooled: bn-pooled (m1 + AdaBN), bn-pooled-soup (soup + AdaBN). ~7 h after chain 1.

Resume: `tools/watch_job.sh --adopt 271991` and `--adopt 297592` (a dead pid is fine to skip), then ONE `tools/watch_job.sh` Monitor. If still running, report progress and stop (park again if >1.5 h remain), logging whichever experiments have finished.

Steps 4-5 per experiment when its models exist: `$(bash tools/python_path.sh) tools/results.py <control> <model>`, check each fold line's buzz logit SD, read docs/judging-results.md, write notes.md Results/Conclusion, delete this HANDOFF.md (commit), delete/trim the idea in main's IDEAS.md, then tools/finish_experiment.sh from main (LOOP.md step 5). Three-draw depth8 mean is 0.507 (ps-depth8 0.515, ps-depth8-repeat 0.509, ps-depth8-r3 0.497); the soup members add three more draws.

## This experiment
Models in .local/worktrees/ps-d8-soup/models/: members ps-d8-soup-m1/m2/m3, ps-d8-soup-ens, ps-d8-soup (soup; the --model for finish_experiment), ps-d8-soup-w25/w50/w75. Compare: soup vs best member (results.py <best member> ps-d8-soup), soup vs ens, wise vs soup; report each member against the three-draw mean too. If chain 2 died (or chain 1 died first so chain 2's wait ended early): from .local/worktrees/ps-d8-soup, once no GPU job runs, `/home/luke/projects/buzzdetect-training/tools/launch_job.sh chain2_b39.log -- bash chain2_b39.sh` (its wait on 271991 passes at once if that pid is dead; finished member folds and scored soup folds are skipped; if chain 1 is unfinished, relaunch chain 1 first and edit chain2's wait pid to the new one). IDEAS #34: delete.
