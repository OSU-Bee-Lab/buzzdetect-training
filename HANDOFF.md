# HANDOFF: yam64h24 (batch 33, experiment 4 of 4; 2 of 4 logged: d1-ens-teacher, yam64h32; yam64h32-s2 and this remain)

Job: the same job as yam64h32-s2: launch_job pid 1075598, log `.local/worktrees/yam64h32-s2/distill.log`
(that worktree's `chain.sh` runs `bash ../yam64h24/chain.sh` once its own repeat is done).
Adopt it: `tools/watch_job.sh --adopt 1075598`, then ONE `tools/watch_job.sh` Monitor (30 min, re-armed).
This part: cache C + V for the new front end (~1 h), pack `C__yam64h24` / `V__yam64h24` shards (~25 GB; / had
44 GB free at launch), WSD trunks to 56000 with the stop rule on `yam64h24:a0.50_d8` and `a0.25_d8`, then mmt
re-export of every judged branch. ~6-7 h after the repeat.

**If it's still running:** report progress (the `[chain] left:` line) and stop (park again if > 1.5 h).

**When it finishes (exit 0):**
1. `python 05_distill/ladder_record.py wsd --name fe_C_yam64h24_a0.50_d8_s1_c-buzz-rain-human_wsd` and the same
   for `a0.25_d8`; judged budget = where the stop rule fired (`dpaths.wsd_stop_index` over hit@K), else 56000.
   `ladder_record.py frontier`, `ladder_record.py repeats`.
2. Comparators (launch_job; `$(bash tools/python_path.sh) tools/results.py <control> <exp>`): the hop-32 twins
   `fe_C_yam64h32_<arch>_s1_c-buzz-rain-human_wsd56000_mmt` and the hop-16 twins `fe_C_yam64h16_<arch>_s1_..._wsd56000_mmt`
   against `fe_C_yam64h24_<arch>_s1_c-buzz-rain-human_wsd<judged>_mmt`, both archs. Where it sits against
   `twofast32:a0.50_d8` (0.694-0.701 at 5.33x) and `yam64h32` (0.658 / 6.81x, 0.687 / 6.06x). Speed from `_mmt` rows.
3. Results/Conclusion in notes.md, remove this HANDOFF.md, commit, then from main
   `tools/finish_experiment.sh yam64h24 --arm distill --summary "..." --runs "<all fe_C_yam64h24_* rows incl. _mmt>"`.
   Then `loop_signal.sh done "d1-ens-teacher yam64h32 yam64h32-s2 yam64h24"`.

**If it died:** read the log tail. Relaunch just this part (resumes every stage):
`cd .local/worktrees/yam64h24 && tools/launch_job.sh distill.log -- bash chain.sh`
(if the repeat is also unfinished, relaunch yam64h32-s2's chain instead; it runs this one after).
