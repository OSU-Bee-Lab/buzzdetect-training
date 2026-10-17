# HANDOFF d2-b64w16 (batch 35)
Batch 35: 1 of 4 logged (d1-ens-yamnet). D2 blocked: plan.csv relpaths for 46 V slices point at renamed audio (250722_1415.mp3 -> 260722_1415.mp3 etc.); Wooster 2025-08-06/1_46/250806_1423.mp3 has no counterpart (Luke's call).
- No job running. Once the plan remap / symlinks exist: `tools/launch_job.sh distill.log -- bash chain.sh` in this worktree (fills V mel, trains, re-exports mmt), one `tools/watch_job.sh` Monitor.
- When done: `ladder_record.py frontier`, `tools/results.py` vs matched `fe_C_yam64h16_a0.50_d8_s1_c-buzz-rain-human_wsd<N>_mmt`, notes Results/Conclusion, delete IDEAS D2, `tools/finish_experiment.sh d2-b64w16 --arm distill --summary ... --runs ...`.
- Next candidates: IDEAS #29 BEATs (big build; check LOOP.md AVES injunction).
