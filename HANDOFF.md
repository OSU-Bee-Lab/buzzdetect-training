# HANDOFF: fast32h16-depth (chain part 2: fast32h16 a0.50_d8)

## Batch 25 state (2026-10-07 17:25)
0 of 4 experiments logged. One chained job runs three of them in sequence: `fast32-d8d6` (fast32 a0.50_d8 then a0.50_d6), then `fast32h16-depth` (fast32h16 a0.50_d8), then `shallow-wide` (fast32h16 a0.75_d8). Experiment 4 is not started (see fast32-d8d6/HANDOFF.md, "Experiment 4").

- **Job:** launch_job pid **3824760**, log `.local/worktrees/fast32-d8d6/distill.log`, script `.local/worktrees/fast32-d8d6/chain.sh`. Adopt with `tools/watch_job.sh --adopt 3824760`, then one `tools/watch_job.sh` Monitor.
- **If it is still running, report progress and stop** (park again with an ETA). Measured: fast32 d8 trunk at ~4.8 step/s; WSD to 56k is ~3.2 h trunk + decays/evals, ~4 h per student unless the stop rule fires; four students ≈ 15 h from 16:50.
- **If it died:** read distill.log. Relaunch the whole chain from the fast32-d8d6 worktree (main.py skips finished students and resumes checkpoints): `cd .local/worktrees/fast32-d8d6 && /home/luke/projects/buzzdetect-training/tools/launch_job.sh distill.log -- ./chain.sh`. The archs (`a0.50_d8`, `a0.50_d6`, `a0.75_d8`) exist only on these branches, so never run from main.
- **When this experiment's students are recorded:** `python 05_distill/ladder_record.py frontier`, `ladder_record.py wsd --name <trunk>`, per-fold tables with `tools/results.py 05_distill/data/v4-ft-ps-e60-moderate/eval/<control> .../eval/<student>` (launch_job it, can exceed 2 min). Fill notes.md Results/Conclusion (headline and speed together, tiers, frontier position), delete this file, commit, then from main `tools/finish_experiment.sh <slug> --arm distill --summary "..." --runs "<every recorded student name>"`. Logging may happen while the chain still runs later experiments: finishing one does not touch the others' worktrees.

- **This experiment's comparators:** fe_C_fast32h16_a0.50_s1_c-buzz-rain-human_wsd* (0.650 @2.50x at 56k); frontier neighbours fast32h16 a0.375 0.639 @2.63x, a0.25 0.612 @2.75x. Student: fe_C_fast32h16_a0.50_d8_s1_c-buzz-rain-human_wsd*.
