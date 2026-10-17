# HANDOFF d1-ens-yamnet (batch 35)
Batch 35 progress: 0 of 4 experiments logged. d2-b64w16 is BLOCKED (renamed audio breaks V pack; friction signalled, notes on exp/d2-b64w16; resume with `tools/launch_job.sh distill.log -- bash chain.sh` in that worktree once paths resolve). This one (d1-ens-yamnet) is running.

- Job: launch_job pid 1937358, log .local/worktrees/d1-ens-yamnet/distill.log. Resume: `tools/watch_job.sh --adopt 1937358`, then one `tools/watch_job.sh` Monitor.
- Training started 09:3x at 4.4 step/s, <=63350 steps in 8 stages (~4 h) + judging. If still running, report progress and park again.
- When it finishes (exit 0): `python 05_distill/ladder_record.py frontier`, `ladder_record.py wsd --name ens3_C_yam64h16_a0.50_d8_s1_c-buzz-rain-human_wsd`, `tools/results.py fe_C_yam64h16_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt ens3_C_yam64h16_..._wsd56000_mmt` (via launch_job). Prediction >0.747, falsifier within 0.02 of 0.727. Write notes.md Results/Conclusion, delete IDEAS.md D1 section if conclusive (D1 remainder), `tools/finish_experiment.sh d1-ens-yamnet --arm distill --runs "<ens3 mmt student names>"`.
- If died: relaunch `tools/launch_job.sh distill.log -- bash chain.sh` from this worktree (resumes). Remaining batch experiments after this: pick from IDEAS (#29 BEATs, or D2 if unblocked).
