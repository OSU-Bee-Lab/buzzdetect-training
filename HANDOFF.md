# HANDOFF: l5x050-yam64h32-s2 (batch 34)

**Batch 34 count: 2 of 4 logged (nonuniform-width, nonuniform-width-yam64h32). This is slot 3, in flight. Slot 4 unchosen
(candidate: `ladder_record.py repeats` also flags `main.py --rung B --seed 2 --wsd 7000,14000,28000,56000 --runs "fast32h16:a0.50:classes=ins_buzz+ambient_rain+human"`, not a wide-late run; or an IDEAS.md item).**

- Job: launch_job pid **340040**, log `/home/luke/projects/buzzdetect-training/.local/worktrees/l5x050-yam64h32-s2/distill.log`
  (`./chain.sh`: seed-2 `yam64h32:a0.25_d8_l5x0.50`, WSD 7000-56000, then mmt re-export). Started 22:42; ETA at 22:56 was ~1h58m plus the re-export.
  Resume: `tools/watch_job.sh --adopt 340040`, then ONE Monitor on `tools/watch_job.sh`.
- If it's still running: report progress (`[chain] left:` line) and stop, parking again if > 1.5 h remain.
- When it finishes (`fe_C_yam64h32_a0.25_d8_l5x0.50_s2_c-buzz-rain-human_wsd<N>_mmt` rows in main's `05_distill/log.jsonl`):
  1. `ladder_record.py frontier`, `ladder_record.py wsd --name fe_C_yam64h32_a0.25_d8_l5x0.50_s2_c-buzz-rain-human_wsd`.
  2. `tools/results.py` (python from tools/python_path.sh, via launch_job, one pair per call): student `..._l5x0.50_s2_..._wsd<last>_mmt` against controls
     `fe_C_yam64h32_a0.25_d8_s2_c-buzz-rain-human_wsd56000_mmt` (same seed), `fe_C_yam64h32_a0.50_d8_s1_..._wsd56000_mmt`, and the seed-1 l5x0.50 row.
  3. Fill notes.md Results/Conclusion against its Prediction/Falsifier, delete HANDOFF.md, commit, then from main:
     `tools/finish_experiment.sh l5x050-yam64h32-s2 --arm distill --summary "..." --runs "<the _mmt student names>"`.
- If it died: read the log's tail, fix, relaunch (resumes): `cd .local/worktrees/l5x050-yam64h32-s2 && tools/launch_job.sh distill.log -- ./chain.sh`
