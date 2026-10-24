# HANDOFF: quietup3-d8 (batch 44)

Batch 44 runs four experiments in one chain: quietup3-d8, quietup6-d8, focal-d8, mixup-d8
(in that order, one medium CV each, ~2h10m per CV, ~16 min per fold). **This worktree's part is `quietup3-d8`.**
None of the four was logged when this was written (2026-10-24 09:50); the training queue's only other item
(IDEAS #29) is blocked on Luke and was not run.

- **Job:** launch_job pid **3147024**, log `.local/worktrees/quietup3-d8/chain_b44.log`
  (script `.local/worktrees/quietup3-d8/chain_b44.sh`; each CV also tees into its own worktree's `train.log`).
  Adopt it: `tools/watch_job.sh --adopt 3147024`, then arm ONE Monitor on `tools/watch_job.sh`.
- **If it's still running:** report progress (which CV, folds done) and stop; park again if over 1.5 h remains.
- **When `models/quietup3-d8/folds_sx.csv` exists:** check each fold's `buzz logit SD` in `train.log`, then
  `$(bash tools/python_path.sh) tools/results.py ps-depth8-r3 quietup3-d8` (control: `ps-depth8-r3`, 0.497; depth8
  three-draw mean 0.507). Read `docs/judging-results.md`, fill notes.md's Results/Conclusion against the
  Hypothesis already there, delete this HANDOFF.md (commit), then LOOP.md step 5:
  `tools/finish_experiment.sh quietup3-d8 --summary ... --model .local/worktrees/quietup3-d8/models/quietup3-d8 --baseline-model ps-depth8-r3 --hypothesis ... --trust ... --conclusion ...`
  (no IDEAS.md change: not from the queue).
- **If it died:** read the end of `chain_b44.log`. Relaunch from `.local/worktrees/quietup3-d8`:
  `tools/launch_job.sh chain_b44.log -- bash chain_b44.sh` (smokes are marked done by `.smoke_ok`, finished
  folds and CVs are skipped, so it resumes where it stopped).
