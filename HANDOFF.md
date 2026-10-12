# HANDOFF: twofast32-d8-s2 (batch 32, experiment 2 of 4)

Batch state: **1 of 4 experiments logged** (`ps-d8-ktails`, 0.524, mixed). This is the second; two remain
after it. Suggested next, from main's IDEAS.md: D2 (`yam64h16` front end), then the first step of D1
(ensemble confirmation on `moderate`, approved by Luke 2026-10-11; `ps-d8-ktails` says train the members as
separate runs, not as K tails in one job).

## The job
- pid **1296765**, log `.local/worktrees/twofast32-d8-s2/distill.log`, launched 2026-10-12 08:53 with
  `tools/launch_job.sh distill.log -- bash chain.sh` from this worktree.
- Adopt it: `tools/watch_job.sh --adopt 1296765`, then one `tools/watch_job.sh` Monitor.
- It is the seed-2 repeat of `twofast32:a0.50_d8` and `twofast32:a0.25_d8` (rung C, WSD 7000 / 14000 /
  28000 / 56000, classes buzz+rain+human), then the `mmt` re-export of the new branches. At 09:23 it was at
  step 5500 of the first trunk at 3.5 step/s, with at most 126700 train steps in 16 train stages left:
  about 10 h of training plus judging, so done around 19:30-20:00. The chain prints `[chain] left: ...`
  with its own estimate once a train stage has finished; the Monitor's state line repeats it.

## If it is still running
Report progress (which stage, the chain's `left:` estimate) and stop: re-park if more than 1.5 h remain.

## When it finishes (`[launch_job] exit 0`)
The rows are already in main's `05_distill/log.jsonl` (written by main.py's record stage and reexport.py).
1. From main: `$(bash tools/python_path.sh) 05_distill/ladder_record.py frontier`, and
   `ladder_record.py wsd --name fe_C_twofast32_a0.50_d8_s2_c-buzz-rain-human_wsd` (and the a0.25 one) for the
   step-budget curves; read the last doubling of a0.50_d8 (seed 1 went 0.687 -> 0.701).
2. `tools/launch_job.sh results.log -- bash results.sh` from this worktree (four `tools/results.py` tables:
   each seed 2 against its seed 1, and against the fast32h16 student its seed-1 gain was read over). Speed
   comes from the `_mmt` rows.
3. Judge against notes.md's falsifier: seed 2 of `a0.50_d8` at or under 0.676, or `a0.25_d8` under 0.666,
   means that gain does not count. Seed-1 figures: a0.50_d8 0.701 at 5.33x, a0.25_d8 0.668 at 6.14x.
4. Write notes.md Results / Conclusion (docs/judging-results.md first), delete this HANDOFF.md, commit, then
   from main: `tools/finish_experiment.sh twofast32-d8-s2 --arm distill --summary "..." --runs "<every
   seed-2 student name, _mmt ones included>"`. Delete IDEAS.md section D3 in main; main's IDEAS.md carries
   other uncommitted edits (Luke's and earlier batches'), so do not pass `--commit-also IDEAS.md`.

## If it died (exit non-zero, or pid gone with no exit 0)
Read the tail of distill.log. main.py resumes at stage and training-checkpoint level. Relaunch from this
worktree: `tools/launch_job.sh distill.log -- bash chain.sh`
