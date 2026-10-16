# HANDOFF: nonuniform-width-yam64h32 (batch 34)

**2026-10-16 17:15: nonuniform-width (twofast32) is LOGGED and finished: 1 of 4 experiments done. The yam64h32 chain is running as
launch_job pid 46447, log `distill.log` in THIS worktree (not ../nonuniform-width). Adopt 46447. ~8 h remain; parked.
Slots 3-4 unchosen (`ladder_record.py repeats` flagged fe_B_fast32h16_a0.50 seed-2 repeat as a candidate).
Note: tools/results.py takes ONE control and ONE student per call (run the four pairs separately), and must be run with the python from tools/python_path.sh.**

**2026-10-16: chain relaunched as launch_job pid 18975 (log in ../nonuniform-width/distill.log); adopt that pid.**

Batch 34's experiment count: 0 of 4 logged when parked. This batch's two experiments are `nonuniform-width`
(twofast32) and `nonuniform-width-yam64h32`, chained in one job; the remaining two slots are unchosen
(candidates: a seed-2 repeat if `ladder_record.py repeats` flags one; IDEAS D2 if root-disk space is freed).

This worktree's part: the yam64h32 pair (yam64h32:a0.25_d8_l5x0.50, yam64h32:a0.25_d8_l7x1.0) and their mmt re-export, the second half of the chain (run as ../nonuniform-width-yam64h32/chain.sh yam64h32 from ../nonuniform-width/chain.sh).

- Job: launch_job pid **1955094**, log `/home/luke/projects/buzzdetect-training/.local/worktrees/nonuniform-width/distill.log`
  (one job for both experiments). Resume: `tools/watch_job.sh --adopt 1955094`, then ONE Monitor on `tools/watch_job.sh`.
- If it's still running: report progress (the `[chain] left:` line) and stop, parking again if > 1.5 h remain.
- When this part finishes (its `_mmt` rows are in `05_distill/log.jsonl`):
  1. `python 05_distill/ladder_record.py frontier`, `ladder_record.py wsd --name <student>_wsd`, `ladder_record.py repeats`.
  2. `tools/results.py <control> <student>` (launch via launch_job: >2 min) for controls fe_C_yam64h32_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt (seed 2: ..._s2_...) and fe_C_yam64h32_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt
     against fe_C_yam64h32_a0.25_d8_l5x0.50_s1_c-buzz-rain-human_wsd<last>_mmt fe_C_yam64h32_a0.25_d8_l7x1.0_s1_c-buzz-rain-human_wsd<last>_mmt (<last> = the last WSD budget recorded, 56000 unless the stop rule fired earlier).
  3. Fill notes.md Results/Conclusion (headline, speed, tiers; falsifier in Hypothesis), delete HANDOFF.md, commit,
     then from main: `tools/finish_experiment.sh nonuniform-width-yam64h32 --arm distill --summary "..." --runs "<the _mmt student names>"`.
- If it died: read the log's tail, fix, and relaunch (main.py and reexport.py resume):
  `if the twofast32 half is logged: cd .local/worktrees/nonuniform-width-yam64h32 && tools/launch_job.sh distill.log -- ./chain.sh yam64h32 ; otherwise relaunch ../nonuniform-width/chain.sh (it chains into this one)`
