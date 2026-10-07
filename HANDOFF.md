# HANDOFF: fast32-depth (batch 24, experiment 4 of 4)

Batch 24 status (2026-10-07 07:45): 3 of 4 logged (ps-depth8-repeat, ps-depth6, ps-depth4). This is the last; when it is logged, send `loop_signal.sh done "ps-depth8-repeat ps-depth6 ps-depth4 fast32-depth"`.

Progress (2026-10-07 14:43, resumed session): d12 finished; d10 28k recorded (headline 0.696, 2.28x); d10 trunk at 26000/47600 (~4.2 step/s), then 47.6k decay, possibly 56k. Parked again, ETA ~2 h.

- Job: launch_job pid **3438117**, log `.local/worktrees/fast32-depth/distill.log`. One `05_distill/main.py` call training two students in sequence, fast32 a0.50_d12 then a0.50_d10 (buzz+rain+human, rung C, WSD to 56k with `--wsd-stop 1.3`). Adopt with `tools/watch_job.sh --adopt 3438117`, then one `tools/watch_job.sh` Monitor.
- **If it's still running, report progress and stop** (park again with an ETA). Measured: the d12 trunk did 5950 steps in 27 min (~0.27 s/step), so up to 56k is ~4 h trunk + ~1 h decays per student, less if the stop rule fires early. Both students: ~6-10 h from 07:13.
- **When it finishes (exit 0):** `python 05_distill/ladder_record.py frontier` and `ladder_record.py wsd --name fe_C_fast32_a0.50_d12_s1_c-buzz-rain-human_wsd` (same for d10). Matched control: fe_C_fast32_a0.50_s1_c-buzz-rain-human_wsd* (0.680/0.687 at 28k/56k, 2.16x). Per-fold tables: `tools/results.py 05_distill/data/v4-ft-ps-e60-moderate/eval/<control> .../eval/<student>`. Say where each lands against the frontier (fast32 a0.375 0.675 @2.28x, fast32h16 a0.50 0.650 @2.51x). Fill notes.md's Results/Conclusion, delete the "dropping layers beyond d12" mention from main's IDEAS.md Untried list, delete this file, commit, then from main: `tools/finish_experiment.sh fast32-depth --arm distill --summary "..." --runs "<every recorded student name>" [--commit-also IDEAS.md]`.
- **If it died:** read distill.log. Relaunch from this worktree with the same command (main.py resumes at stage and checkpoint level):
  `tools/launch_job.sh distill.log -- 05_distill/main.py --rung C --runs "fast32:a0.50_d12::classes=ins_buzz+ambient_rain+human fast32:a0.50_d10::classes=ins_buzz+ambient_rain+human" --wsd-max 56000 --wsd-stop 1.3`
  Note `a0.50_d10` exists only on this branch (`distill_train.ARCHS`), so it must run from this worktree.
