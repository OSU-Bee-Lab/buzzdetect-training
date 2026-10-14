# HANDOFF: yam64h32 (batch 33, experiment 2 of 4; 1 of 4 logged: d1-ens-teacher; this one and two more remain)

Job: launch_job pid 4107621, log `.local/worktrees/yam64h32/distill.log` (a wait-wrapper that ran `chain.sh` once
the D1 chain exited). Adopt it: `tools/watch_job.sh --adopt 4107621`, then ONE `tools/watch_job.sh` Monitor
(30 min, re-armed).

Done already: bench (`bench.log`, table in notes.md: a0.25_d8 6.70x, a0.50_d8 6.00x). At park time (08:50 on
10-14) the chain was in `cache C` for the new front end (4455/48395 slices at 12.4/s, ~1 h), then packs
`C__yam64h32` / `V__yam64h32` shards (~17 GB + ~2 GB; / had 70 GB free) and trains two WSD trunks to 56000
(`yam64h32:a0.25_d8`, `a0.50_d8`), then the `mmt` re-export of every judged branch. Total ~8 h.

**If it's still running:** report progress (the `[chain] left:` line) and stop (park again if > 1.5 h).

**When it finishes (exit 0):**
1. `python 05_distill/ladder_record.py wsd --name fe_C_yam64h32_a0.25_d8_s1_c-buzz-rain-human_wsd` and the same
   for `a0.50_d8`; the judged budget is the last one the stop rule reached. `ladder_record.py frontier` for position.
2. Comparators (launch through launch_job, can run past 2 min), the hop-16 twins:
   `tools/results.py fe_C_yam64h16_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt fe_C_yam64h32_a0.25_d8_s1_c-buzz-rain-human_wsd<judged>_mmt`
   and the same for `a0.50_d8`. Also say where each sits against `twofast32:a0.50_d8` (0.694-0.701 at 5.33x),
   `twofast32:a0.25_d8` (0.668 at ~6.1x) and `fast32h16` (0.646 at 6.7x). Speed from the `_mmt` rows.
   Repeat rule: `ladder_record.py repeats` (a gain > 0.02 over everything at its speed or faster gets a seed-2 repeat;
   that repeat can be one of the remaining experiments).
3. LOOP.md steps 4-5: Results/Conclusion in notes.md, remove this HANDOFF.md, then from main
   `tools/finish_experiment.sh yam64h32 --arm distill --summary "..." --runs "<all fe_C_yam64h32_* rows incl. _mmt>"`.
   No IDEAS.md entry to delete (the lead came from yam64h16's notes).

**If it died:** read the log's tail. Relaunch (resumes every stage, training from its checkpoint):
`cd .local/worktrees/yam64h32 && tools/launch_job.sh distill.log -- bash chain.sh`
(chain.sh reruns the ~3 min bench first; harmless.)

**Remaining after this:** two more experiments. Candidate leads: if yam64h32 holds its hop-16 twins' headline, 48 bands
at hop 32 (`yam48h32`) or a cheaper 64-band STFT (the 64-band mel front end, 6.6x alone, is now the speed
ceiling); IDEAS D1 remainder (ensemble teacher through a wide student, ~34 GB more shards).
