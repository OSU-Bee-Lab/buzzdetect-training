# HANDOFF: fast32h16-a050-s2 (batch 34)

**Batch 34 count: 3 of 4 logged (nonuniform-width, nonuniform-width-yam64h32, l5x050-yam64h32-s2). This is slot 4, in flight; nothing else remains after it.**

- Job: launch_job pid **551239**, log `/home/luke/projects/buzzdetect-training/.local/worktrees/fast32h16-a050-s2/distill.log` (`./chain.sh`: rung B seed 2, `fast32h16:a0.50:classes=ins_buzz+ambient_rain+human`, WSD 7000-56000). ETA at 01:46 was ~3h50m.
  Resume: `tools/watch_job.sh --adopt 551239`, then ONE Monitor on `tools/watch_job.sh`.
- If still running: report progress (`[chain] left:` line); park again if > 1.5 h remain.
- When it finishes (rows `fe_B_fast32h16_a0.50_s2_c-buzz-rain-human_wsd<N>` in main's `05_distill/log.jsonl`; no mmt re-export in chain.sh, rung B has none; judge by those rows):
  1. `ladder_record.py frontier`, `ladder_record.py wsd --name fe_B_fast32h16_a0.50_s2_c-buzz-rain-human_wsd`.
  2. `tools/results.py` (python via tools/python_path.sh): `fe_B_fast32h16_a0.50_s2_..._wsd56000` against `fe_B_fast32h16_a0.50_s1_c-buzz-rain-human_wsd56000` (0.588).
  3. Fill notes.md Results/Conclusion against Prediction/Falsifier, delete HANDOFF.md, commit, then from main:
     `tools/finish_experiment.sh fast32h16-a050-s2 --arm distill --summary "..." --runs "<student names>"`.
- If it died: read the log tail, fix, relaunch (resumes): `cd .local/worktrees/fast32h16-a050-s2 && tools/launch_job.sh distill.log -- ./chain.sh`
