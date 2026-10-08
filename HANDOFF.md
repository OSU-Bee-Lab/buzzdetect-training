# HANDOFF: yamnet-depth (batch 25, experiment 4 of 4)

## Batch 25 state (2026-10-08 07:40; re-parked 12:40)

12:40 resume: still running, trunk step 35000/47600 at 3.2 step/s (~2000 steps/10 min). wsd28000 already recorded (headline 0.735 @1.52x/1.76x). Remaining: trunk ~63 min + 47600->56000 decay/probe/speed ~45 min (28k decay took 2410 s) -> ETA ~14:30. Parked again.

3 of 4 experiments logged (fast32-d8d6, fast32h16-depth, shallow-wide). This is the 4th and last; once it is logged, send `loop_signal.sh done "fast32-d8d6 fast32h16-depth shallow-wide yamnet-depth"`.

- **Job:** launch_job pid **145616**, log `.local/worktrees/yamnet-depth/distill.log`. Adopt with `tools/watch_job.sh --adopt 145616`, then one `tools/watch_job.sh` Monitor.
- **If it's still running, report progress and stop** (park again with an ETA). Measured: 3.1 step/s on the trunk (07:35, step 2500/47600), so ~4 h trunk + decays/evals ≈ 5 h from 07:40 unless the stop rule fires.
- **If it died:** read distill.log. Relaunch from this worktree (main.py resumes at stage/checkpoint level; arch `a0.50_d8` exists only on this branch, never run from main):
  `cd /home/luke/projects/buzzdetect-training/.local/worktrees/yamnet-depth && /home/luke/projects/buzzdetect-training/tools/launch_job.sh distill.log -- 05_distill/main.py --rung C --runs "yamnet:a0.50_d8:select:classes=ins_buzz+ambient_rain+human" --wsd-max 56000 --wsd-stop 1.3`
- **When it finishes:** `python 05_distill/ladder_record.py frontier`, `ladder_record.py wsd --name <trunk>`, then (via launch_job) `tools/results.py 05_distill/data/v4-ft-ps-e60-moderate/eval/fe_C_yamnet_a0.50_s1_select_c-buzz-rain-human_wsd28000 05_distill/data/v4-ft-ps-e60-moderate/eval/<student>` and the same against `fe_C_yamnet_a0.25_s1_select_c-buzz-rain-human_wsd56000`. Judge against notes.md's revised hypothesis (gain: > ~0.717 at ~1.50x, the a0.50->a0.25 line). Fill Results/Conclusion, delete this file, commit, then from main `tools/finish_experiment.sh yamnet-depth --arm distill --summary "..." --runs "<every recorded student name>"`.
- **Comparators:** fe_C_yamnet_a0.50_s1_select_c-buzz-rain-human_wsd* (0.725 @1.40x, stop at 28k), fe_C_yamnet_a0.25_s1_select_c-buzz-rain-human_wsd* (0.707 @1.62x). Student: fe_C_yamnet_a0.50_d8_s1_select_c-buzz-rain-human_wsd*.
