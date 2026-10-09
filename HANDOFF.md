# HANDOFF (batch 26, parked 2026-10-08 ~15:30)

**Update 2026-10-09 ~10:45: 3 of 4 logged; only this worktree's a0.75_d8 student remains, now running alone.**
The chain (pid 399369) died at 10:17 with an OOM at step 0 (notes.md, "Run log"). It was fixed on
exp/yamnet-d8-repeat (a56376c, val batch 128) and relaunched alone: **pid 1377627**, log
`.local/worktrees/yamnet-d8-repeat/distill.log`, started 10:28; at 10:55 it was at train 3500/47600, 2.3 step/s (past the step-0 val that OOMed); ETA ~18:00. Resume:
`tools/watch_job.sh --adopt 1377627`. The pid 399369 and relaunch-command bullets below are superseded: if it
died, relaunch from `.local/worktrees/yamnet-d8-repeat` (it resumes from its last checkpoint):
```
S="classes=ins_buzz+ambient_rain+human"; W="--rung C --wsd-max 56000 --wsd-stop 1.3"
/home/luke/projects/buzzdetect-training/tools/launch_job.sh distill.log -- bash -c "python 05_distill/main.py $W --runs 'yamnet:a0.75_d8:select:$S'"
```
If it OOMs again *in training* (not val), the 4 GB card can't hold a0.75_d8 at batch 512 next to whatever else is
resident; write that up as the result (width beyond a0.50 not trainable on this card at the recipe's batch) rather
than change the batch.

One job runs all four of batch 26's distillation experiments, in order, from the
`yamnet-d8-repeat` worktree (the only branch with the `parse_arch` fix in
`05_distill/distill_train.py`; the other three worktrees lack it, so never launch from them):

1. `yamnet-d8-repeat`: yamnet a0.50_d8 **seed 2** (repeat of yamnet-depth)
2. `yamnet-a0375-d8`: yamnet a0.375_d8, seed 1
3. `yamnet-a025-d8`: yamnet a0.25_d8, seed 1
4. `yamnet-a075-d8`: yamnet a0.75_d8, seed 1

All rung C, classes buzz+rain+human, select init, WSD to 56k with --wsd-stop 1.3.
**Update 2026-10-09 10:15 (re-park): 3 of 4 logged** (`yamnet-d8-repeat`, `yamnet-a0375-d8` 0.716 @1.58x, `yamnet-a025-d8` 0.713 @1.69x; only this worktree's student remains). a0.75_d8 started 10:03 2026-10-09; expect ~8 h, done ~18:00-19:00. Earlier update 2026-10-08 22:40: 1 of 4 logged (`yamnet-d8-repeat` done: seed 2 0.725 @1.51x, stopped at 28k; use the **two-seed a0.50_d8 mean 0.730** as the a0.50_d8 comparator point, not seed 1's 0.735; eval dirs for both seeds exist). At 22:26 the chain was on student 2 (a0.375_d8) at train step 20500/47600; student 1 took 4.3 h wall. **This worktree's part: student 4 (a0.75_d8), last in the chain; comparator: a0.50_d8 s1 wsd56000 (0.735 @1.52x; two-seed mean 0.730), and full-depth yamnet a0.75 if logged.** After recording it, all 4 are done: send `loop_signal.sh done "yamnet-d8-repeat yamnet-a0375-d8 yamnet-a025-d8 yamnet-a075-d8"`.

- **pid 399369**, log `.local/worktrees/yamnet-d8-repeat/distill.log`.
  Resume: `tools/watch_job.sh --adopt 399369`, then one `tools/watch_job.sh` Monitor (30 min, re-arm).
- Measured: 3.3 step/s; yamnet-depth's identical seed-1 run took 7.6 h wall for the full curve.
  Expect ~7-8 h per student (a0.75 slower, a0.25 faster): ~28-30 h total from 14:55 2026-10-08.
- **If it's still running, report progress and stop** (park again with the ETA).
- **When this worktree's student finishes** (its `_wsd<N>` rows appear in `05_distill/log.jsonl`
  and the run's `curve: done` line is in distill.log): read results per LOOP.md step 4 against
  the comparator named in `notes.md`'s Hypothesis (`ladder_record.py frontier`,
  `ladder_record.py wsd --name <trunk>`, `tools/results.py <control eval dir> <student eval dir>`
  via launch_job), write Results/Conclusion in notes.md, delete this HANDOFF.md, then
  `tools/finish_experiment.sh <slug> --arm distill --summary "..." --runs "<recorded student name(s)>"`
  (steps 4-5). Students are named `fe_C_yamnet_<arch>_s<seed>_select_c-buzz-rain-human_wsd<steps>`.
- **If it died:** read the end of distill.log. Relaunch from `.local/worktrees/yamnet-d8-repeat`
  (stages resume; finished students are skipped):
  ```
  S="classes=ins_buzz+ambient_rain+human"; W="--rung C --wsd-max 56000 --wsd-stop 1.3"
  /home/luke/projects/buzzdetect-training/tools/launch_job.sh distill.log -- bash -c "python 05_distill/main.py $W --seed 2 --runs 'yamnet:a0.50_d8:select:$S' && python 05_distill/main.py $W --runs 'yamnet:a0.375_d8:select:$S yamnet:a0.25_d8:select:$S yamnet:a0.75_d8:select:$S'"
  ```
