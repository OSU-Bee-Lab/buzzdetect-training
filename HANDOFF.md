# HANDOFF: yam64h32-s2 (batch 33, experiment 3 of 4; 2 of 4 logged: d1-ens-teacher, yam64h32; this and yam64h24 remain)

Job: launch_job pid 1075598, log `.local/worktrees/yam64h32-s2/distill.log`. ONE job runs both remaining
experiments: `chain.sh` here (seed-2 repeat of `yam64h32:a0.25_d8`, WSD 7000-56000, then mmt re-export of its
s2 branches), then `bash ../yam64h24/chain.sh` (experiment 4, see that worktree's HANDOFF.md).
Adopt it: `tools/watch_job.sh --adopt 1075598`, then ONE `tools/watch_job.sh` Monitor (30 min, re-armed).
At park (16:53 on 10-14) the repeat had ~1h52m left; yam64h24 then needs a new mel cache, shards and two
trunks (~6-7 h).

**If it's still running:** report progress (the `[chain] left:` line) and stop (park again if > 1.5 h).
This experiment is done once `distill.log` shows `[chain] yam64h32-s2 done; starting yam64h24`; finish it then,
without waiting for yam64h24 (but never edit files under ../yam64h24 while its chain runs).

**When its part is done:**
1. `python 05_distill/ladder_record.py wsd --name fe_C_yam64h32_a0.25_d8_s2_c-buzz-rain-human_wsd`; judged budget
   is 56000 (the seed-1 budgets). `ladder_record.py frontier`.
2. Comparator (launch_job; results.py needs the interpreter: `$(bash tools/python_path.sh) tools/results.py`):
   `fe_C_yam64h32_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt fe_C_yam64h32_a0.25_d8_s2_c-buzz-rain-human_wsd56000_mmt`.
   Falsifier in notes.md: seed 2 at or under 0.650. Also against `fast32h16:a0.25_d8_f2` (0.630 at 6.83x).
3. Results/Conclusion in notes.md, remove this HANDOFF.md, commit, then from main
   `tools/finish_experiment.sh yam64h32-s2 --arm distill --summary "..." --runs "<all fe_C_yam64h32_a0.25_d8_s2_* rows incl. _mmt>"`.

**If it died:** read the log tail. Relaunch (resumes every stage, training from its checkpoint; also runs yam64h24 after):
`cd .local/worktrees/yam64h32-s2 && tools/launch_job.sh distill.log -- bash chain.sh`
