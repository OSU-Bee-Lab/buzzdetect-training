# HANDOFF pauc-l01
## Batch 39 state (updated 18:55 2026-10-20, third park)
4 experiments, 3 logged: mixstyle-xfold (a849067), ps-d8-soup (4407d63), bn-pooled (32648df). Remaining: pauc-l01 only.
- chain 2 finished (exit 0, 18:25).
- chain 3: launch_job pid 429588, log .local/worktrees/pauc-l01/chain3_b39.log (`pauc-l01/chain3_b39.sh`, execs mixstyle-xfold/chain_b39.sh): lite smoke passed 18:27 (test_pauc-l01_smoke 0.247), medium CV of pauc-l01 started ~18:27 (~2.2 h). ETA ~20:40.

Resume: `tools/watch_job.sh --adopt 429588` (skip if dead), then ONE `tools/watch_job.sh` Monitor. If still running, report progress and stop (park again if >1.5 h remain).

Steps 4-5 per experiment when its models exist: `$(bash tools/python_path.sh) tools/results.py <control> <model>`, check each fold line's buzz logit SD, read docs/judging-results.md, write notes.md Results/Conclusion, delete this HANDOFF.md (commit only notes.md/HANDOFF.md: `git commit -a` in a worktree also commits the embedders/ symlink swap), delete/trim the idea in main's IDEAS.md, then tools/finish_experiment.sh from main (LOOP.md step 5). ps-depth8-r3 lives at .local/worktrees/ps-depth8-redraw/models/ps-depth8-r3 (pass that path as --baseline-model). Three-draw depth8 mean is 0.507 (ps-depth8 0.515, ps-depth8-repeat 0.509, ps-depth8-r3 0.497); the soup members add three more draws.

## This experiment
Model .local/worktrees/pauc-l01/models/pauc-l01 (step 2 of chain 1). Controls: ps-depth8-r3 (matched), pauc-l03 (0.406) for the dose ladder, three-draw mean 0.507. IDEAS #33: delete the section (last rung). If chain 3 died: once no GPU job runs, from .local/worktrees/pauc-l01: `/home/luke/projects/buzzdetect-training/tools/launch_job.sh chain3_b39.log -- bash chain3_b39.sh` (its wait passes at once when chain 2 is gone; finished folds resume).
