# HANDOFF: yam64h16sp2 (batch 37, experiment 1 of 4; 0 of 4 logged, 3 remain after this one)
Job: pid 2129404 (launch_job.sh, `bash chain.sh`), log `.local/worktrees/yam64h16sp2/distill.log`.
Resume: `tools/watch_job.sh --adopt 2129404`, then one `tools/watch_job.sh` Monitor (30 min, re-arm).
If it's still running, report progress and stop (park again if > 1.5 h remain).
When it finishes (exit 0): chain.sh has trained, mmtz-re-exported and same-session timed both students
(`.local/bench/results_{1,2}.json` in this worktree). Comparators: `fe_C_yam64h16_{a0.25_d8,a0.50_d8}_s1_c-buzz-rain-human_wsd56000_mmtz`
vs the `fe_C_yam64h16sp2_..._wsd56000_mmtz` rows. Run `ladder_record.py frontier`, `tools/results.py <twin> <sp2>` per width
(via launch_job), fill notes.md Results/Conclusion against the Hypothesis falsifier, then LOOP.md step 5
(`finish_experiment.sh yam64h16sp2 --arm distill --runs "<sp2 _mmtz rows>"`).
If it died: read distill.log; relaunch from this worktree with `tools/launch_job.sh distill.log -- bash chain.sh`
(cache_fe and main.py resume; re-export skips done rows).

Progress 2026-10-19 02:47 (batch 37 resume): still running; chain says at most 120750 train steps in 15 stages left, ~29h48m (WSD stop may cut it short). Parked again.
