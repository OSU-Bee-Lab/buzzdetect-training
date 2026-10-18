# HANDOFF wsd112-yam64h32 (batch 36; experiment 1 of 4, none logged yet)
- Job: launch_job pid 4043434, log .local/worktrees/wsd112-yam64h32/chain.log (chain.sh: main.py wsd to 112000, then reexport mmt).
  Resume: `tools/watch_job.sh --adopt 4043434`, then the one `tools/watch_job.sh` Monitor.
- If still running: report progress and stop (park again if >1.5h left).
- When finished: compare fe_C_yam64h32_a0.25_d8_s1_c-buzz-rain-human_wsd112000_mmt against the wsd56000_mmt s1 row (0.658 at 6.81x)
  via `ladder_record.py frontier`/`wsd --name ..._wsd` and `tools/results.py` (launch_job). Fill notes.md Results/Conclusion (prediction 0.665+-0.015, falsifier >=0.68,
  seed 2 at 56k was 0.676), then `tools/finish_experiment.sh wsd112-yam64h32 --arm distill --summary ... --runs "<112000_mmt row name>"`; delete HANDOFF.md first.
- If died: rerun in the worktree `tools/launch_job.sh chain.log -- bash chain.sh` (resumes from checkpoint).
- Then experiments 2-4 remain (pick from distill ideas; IDEAS.md D1 is stale: d1-ens-yamnet already logged).
