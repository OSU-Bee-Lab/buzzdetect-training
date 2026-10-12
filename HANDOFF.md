# HANDOFF: ps-d8-ens-same5 (batch 31, experiment 4 of 4)

**Batch state:** 3 of 4 experiments are logged (`ps-d8-ensemble` in `03_train/log.jsonl`; `matmul-stft` and
`twofast32-d8` in `05_distill/log.jsonl`). This is the fourth and last: when it is logged, signal
`loop_signal.sh done "ps-d8-ensemble matmul-stft twofast32-d8 ps-d8-ens-same5"`.

## The job
- pid **538693** (launch_job's own pid), log `.local/worktrees/ps-d8-ens-same5/train.log`, launched
  2026-10-11 21:34 from this worktree: `tools/launch_job.sh train.log -- bash chain.sh`.
- `chain.sh` trains two plain `ps-depth8` draws on `medium`, one after the other: `ps-depth8-r4`, then
  `ps-depth8-r5` (8 rotating folds each, ~16 min a fold, ~2 h 10 min a draw; expect the end around 02:00
  on 2026-10-12). No shipped model is trained.

Adopt it: `tools/watch_job.sh --adopt 538693`, then the one `tools/watch_job.sh` Monitor.

## If it is still running
Report progress (the Monitor's state line counts folds done; 16 in all, the count restarts for the second
draw's log section) and stop: park again if more than 1.5 h remain, otherwise keep the Monitor armed
until `[launch_job] exit 0`.

## When it finishes (exit 0)
1. Check each fold line's `buzz logit SD` in `train.log` for both draws (0.57-1.21 is this recipe's normal
   range; `1_150` sits lowest; near 0.1 is a collapse).
2. Single-draw check: `$(bash tools/python_path.sh) tools/results.py ps-depth8-r3 ps-depth8-r4` and the same
   for `ps-depth8-r5`. Each should land with the other draws (0.515 / 0.509 / 0.497).
3. Ensembles, from this worktree (`W=/home/luke/projects/buzzdetect-training/.local/worktrees`; seconds each):
   `$(bash tools/python_path.sh) tools_exp/ensemble_predictions.py <name> <member dir> ...` with members
   `$W/ps-depth8/models/ps-depth8` (a), `$W/ps-depth8-repeat/models/ps-depth8-repeat` (b),
   `$W/ps-depth8-redraw/models/ps-depth8-r3` (c), `models/ps-depth8-r4` (d), `models/ps-depth8-r5` (e).
   Write `ps-d8-same5` (all five: the hypothesis), the five 4-subsets (`ps-d8-same4-<letters>`) and the ten
   3-subsets (`ps-d8-same3-<letters>`; `abc` is the existing `ps-d8-ens3`, 0.558). Report the mean and
   range of the headline at 3, 4 and 5 members.
4. Comparators: `tools/results.py ps-d8-ens3 ps-d8-same5` and `tools/results.py ps-d8-ens5 ps-d8-same5`
   (both controls are in `$W/ps-d8-ensemble/models/`; bare names resolve). The test, set before training
   (notes.md): same-recipe five above 0.558 and within 0.02 of the mixed-depth five (0.575) means member
   count is what the mixed five bought; at the three's level means the curve is flat and depth diversity
   did it; more than 0.02 under the mixed five but above the three means both.
5. Steps 4-5 of LOOP.md: read `docs/judging-results.md`, finish `notes.md` (Changes, Results, Conclusion:
   headline against its SD, folds against theirs, tiers). In main's `IDEAS.md`, trim item 30 to its part
   (b) (K tails on a shared stem); main's IDEAS.md also carries uncommitted edits that are not this
   experiment's (Luke's approvals in D1 and item 27, and D3 queued by `twofast32-d8`), so do NOT pass
   `--commit-also IDEAS.md`. `git rm HANDOFF.md`, commit, then from main:
   `tools/finish_experiment.sh ps-d8-ens-same5 --summary "..." --model .local/worktrees/ps-d8-ens-same5/models/ps-d8-same5 --baseline-model ps-d8-ens3 --hypothesis "..." --trust <trust> --conclusion "..."`.

## If it died
Read the tail of `train.log`. Training resumes (finished folds are skipped, a finished draw is skipped
whole), so relaunch the same command from this worktree:

    cd .local/worktrees/ps-d8-ens-same5 && tools/launch_job.sh train.log -- bash chain.sh

An out-of-memory after a fold's training finished is an unbatched scoring call (CLAUDE.md), not a reason
for `--cpu`. Nothing else may use the GPU while it runs.
