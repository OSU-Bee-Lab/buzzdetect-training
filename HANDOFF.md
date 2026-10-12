# HANDOFF: ps-d8-ktails (batch 32, experiment 1 of 4)

Batch state: **0 of 4 experiments logged.** This is the first; three remain after it. Suggested next,
from main's IDEAS.md: D3 (seed-2 repeats of the two twofast32 d8 students, owed by the repeat rule; no code,
`exp/twofast32-d8`'s `chain.sh` with `--seed 2 --wsd 7000,14000,28000,56000`), then D2 (`yam64h16`), then
the first step of D1 (ensemble confirmation on `moderate`, approved by Luke 2026-10-11).

## The job
- pid **870804**, log `.local/worktrees/ps-d8-ktails/train.log`, launched 2026-10-12 02:38 with
  `tools/launch_job.sh train.log -- bash chain.sh` from this worktree.
- Adopt it: `tools/watch_job.sh --adopt 870804`, then one `tools/watch_job.sh` Monitor.
- It is one medium CV (8 folds, 30 epochs) of three ps-depth8 tails trained in one process
  (`TRUNK_KTAILS=3`, see notes.md). No per-epoch lines are printed; the first fold line had not appeared at
  03:08. Estimate from the lite smoke (~1.1 s per 3-tail step): ~40 min per fold, done ~07:50.
  A single ps-depth8 draw takes ~16 min per fold: record the measured per-fold time in notes.md.
- `chain.sh` retries the CV up to 4 times (`[chain] attempt N failed; resuming`): the smoke run hit one GPU
  OOM between folds and resumed cleanly. Count the retries and say so in notes.md.

## If it is still running
Report progress (folds done, time per fold) and stop: re-park if more than 1.5 h remain.

## When it finishes (`[launch_job] exit 0`)
Outputs: `models/ps-d8-ktails3` (the 3-tail logit mean) and `models/ps-d8-ktails3-t{0,1,2}` (each tail
alone; `chain.sh` runs `resummarize.py` on them).
1. Check each fold line's `buzz logit SD` in train.log (that is the mean's; 0.57-1.34 in the single draws).
2. Comparators (all in main's or the worktrees' `models/`):
   - `$(bash tools/python_path.sh) tools/results.py ps-d8-ens3 ps-d8-ktails3` (the logged three separate
     draws, 0.558; note it was the best of ten same-recipe threes, mean 0.545, range 0.535-0.558,
     `models/ps-d8-same3-*` in `.local/worktrees/ps-d8-ens-same5`).
   - `tools/results.py ps-depth8-r3 ps-d8-ktails3` (the median single draw, 0.497) and the three
     `-t<k>` headlines against the five singles (0.464-0.515, mean 0.492).
   - Member correlation, to set against the separate draws' 0.62 (negatives) / 0.88 (buzz): the
     correlation of `activation_ins_buzz` between `-t<k>` predictions, split by `correct`, pooled over
     folds (`exp/ps-d8-ensemble`'s notes say how theirs was computed; match it).
   - Wall-clock per fold against a single draw's (~16 min).
3. Write notes.md Results / Conclusion (docs/judging-results.md first), delete this HANDOFF.md, then from
   main: `tools/finish_experiment.sh ps-d8-ktails --summary ... --model
   .local/worktrees/ps-d8-ktails/models/ps-d8-ktails3 --baseline-model ps-d8-ens3 --hypothesis ...
   --trust ... --conclusion ...`. Trim IDEAS.md item 30 in main (it is answered either way); main's
   IDEAS.md already carried uncommitted edits before this batch, so check `git diff IDEAS.md` before
   passing `--commit-also`.

## If it died (exit non-zero, or pid gone with no exit 0)
Read the tail of train.log. Finished folds are kept and skipped on rerun. Relaunch from this worktree:
`tools/launch_job.sh train.log -- bash chain.sh`
If it dies of GPU OOM in a training step on every attempt, lower the load in `chain.sh`
(`TRUNK_BATCH=512 TRUNK_ACCUM=2`, each tail then gets batch 1024's update) and say so in notes.md.
