# HANDOFF yam64h20-cliff (batch 36; experiment 2 of 4. Logged so far: wsd112-yam64h32. Remaining after this: 2 more)
- Job: launch_job pid 17861, log .local/worktrees/yam64h20-cliff/distill.log (chain.sh: cache_fe C+V yam64h20,
  main.py rung C WSD 56000 for yam64h20:a0.25_d8 and a0.50_d8 (buzz+rain+human), then reexport mmt).
  Resume: `tools/watch_job.sh --adopt 17861`, then the one `tools/watch_job.sh` Monitor.
- If still running: report progress (the `[chain] left:` ETA) and stop; park again if >1.5h left.
- When finished: compare the `_mmt` rows of fe_C_yam64h20_{a0.25_d8,a0.50_d8}_s1_c-buzz-rain-human_wsd<last> against the
  yam64h16 and yam64h24 twins (same arch, s1, wsd56000_mmt; h16 0.711@4.57x / 0.727@3.77x, h24 0.658@6.69x / 0.689@5.59x)
  and the frontier (twofast32 a0.50_d8 0.701@5.33x, a0.25_d8 0.668@~6x): `05_distill/ladder_record.py frontier`,
  `wsd --name <trunk>_wsd`, and `tools/results.py <h24 twin> <h20 student>` via launch_job. Fill notes.md Results/Conclusion
  (prediction a0.25_d8 0.685+-0.015, a0.50_d8 0.705+-0.015; falsifier a0.25_d8 <= 0.668). Check the repeat rule
  (`ladder_record.py repeats`). Delete HANDOFF.md, commit, then from main:
  `tools/finish_experiment.sh yam64h20-cliff --arm distill --summary "..." --runs "<both _mmt row names>"`.
- If died: from this worktree `/home/luke/projects/buzzdetect-training/tools/launch_job.sh distill.log -- bash chain.sh`
  (cache_fe and main.py resume).
- Note: .local/worktrees/yam64h20 (exp/yam64h20) is an unrelated empty abandoned start from 2026-10-15; ignore it.
