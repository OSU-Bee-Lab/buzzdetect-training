# HANDOFF: yam64h16 (batch 32, experiment 3 of 4)

Batch state: **2 of 4 experiments logged** (`ps-d8-ktails`, `twofast32-d8-s2`). This is the third; one
remains after it. Suggested fourth, from main's IDEAS.md: the first step of D1 (ensemble confirmation on
`moderate`, approved by Luke 2026-10-11; `ps-d8-ktails` says train the members as separate runs).

## The job
- pid **1511713**, log `.local/worktrees/yam64h16/distill.log`, launched 2026-10-12 18:57 with
  `tools/launch_job.sh distill.log -- bash chain.sh` from this worktree.
- Adopt it: `tools/watch_job.sh --adopt 1511713`, then one `tools/watch_job.sh` Monitor.
- It distils `yam64h16:a0.25_d8` and `yam64h16:a0.50_d8` (rung C, seed 1, `--wsd-max 56000 --wsd-stop 1.3`,
  classes buzz+rain+human), then the `mmt` re-export of the judged branches. At 19:27 it was still in
  `cache_fe C yam64h16` (11549/48395 slices at 10.7/s, done about 20:25); then a rung-C pack and at most
  ~127000 train steps in 16 train stages (the twofast32 pair took ~10.5 h; 64 bands may be slower, and the
  stop rule may end a trunk early). Expect the end between 06:00 and 09:00 on 2026-10-13. The chain prints
  `[chain] left: ...` once a train stage has finished; the Monitor's state line repeats it.

## If it is still running
Report progress (which stage, the chain's `left:` estimate) and stop: re-park if more than 1.5 h remain.

## When it finishes (`[launch_job] exit 0`)
The rows are already in main's `05_distill/log.jsonl`.
1. From main: `$(bash tools/python_path.sh) 05_distill/ladder_record.py frontier`, and
   `ladder_record.py wsd --name fe_C_yam64h16_a0.25_d8_s1_c-buzz-rain-human_wsd` (and the a0.50 one).
   If the stop rule ended a trunk before 56000, edit `results.sh`'s `T` for that student to its last budget.
2. `tools/launch_job.sh results.log -- bash results.sh` from this worktree (each student against its
   10 ms-hop `yamnet` twin, and against `twofast32:a0.50_d8`). Speed comes from the `_mmt` rows.
3. Judge against notes.md: comparators are `yamnet:a0.25_d8` 0.713 at 3.24x, `yamnet:a0.50_d8` 0.735 at
   ~2.6x, and the frontier student `twofast32:a0.50_d8` 0.694-0.701 at 5.33x, which is faster than every
   `yam64h16` graph (bench: a0.25_d8 4.66x, a0.50_d8 3.74x). A `yam64h16` student is a gain only if its
   headline beats every student at its speed or faster (so > ~0.70, by > 0.02 to need a repeat); also
   report the hop cost on 64 bands (yam64h16 minus its yamnet twin; it was ~0.035 on fast32).
4. Write notes.md Results / Conclusion (docs/judging-results.md first), delete this HANDOFF.md, commit,
   delete IDEAS.md section D2 in main, then from main: `tools/finish_experiment.sh yam64h16 --arm distill
   --summary "..." --runs "<every fe_C_yam64h16_* student name, _mmt ones included>" --commit-also IDEAS.md`
   (only if `git diff IDEAS.md` shows nothing but that deletion).

## If it died (exit non-zero, or pid gone with no exit 0)
Read the tail of distill.log. main.py resumes at stage and training-checkpoint level; cache_fe resumes too.
Check free space on `/` first (146 GB free at launch; the rung-C pack needs ~30 GB). Relaunch from this
worktree: `tools/launch_job.sh distill.log -- bash chain.sh`
