# HANDOFF: d1-ens-teacher (batch 33, experiment 1 of 4; 0 of 4 logged, all four remain)

Job: launch_job pid 3229960, log `.local/worktrees/d1-ens-teacher/distill.log` (`chain.sh`).
Adopt it: `tools/watch_job.sh --adopt 3229960`, then ONE `tools/watch_job.sh` Monitor (30 min, re-armed).

Done already: r2/r3 shipped and exported (`chain_ship.sh`, `.local/members/`), ensemble teacher
`v4-ft-ps-e60-moderate-ens3` built by `tools_exp/build_ens_teacher.py` (buzzdetect engine/models + main's
models/; plan.csv and blacklist.txt copied from the single teacher's manifest so slices and V pool are identical).
At park time (01:19 on 10-14) the chain was in `cache C` (9206/48395 slices at 9.6/s), then packs twofast32
shards and trains the WSD trunk to 56000 (~4-5 h).

**If it's still running:** report progress (the `[chain] left:` line) and stop (park again if > 1.5 h).

**When it finishes (exit 0):**
1. `python 05_distill/ladder_record.py wsd --name ens3_C_twofast32_a0.50_d8_s1_c-buzz-rain-human_wsd`
   (with `DISTILL_TEACHER=v4-ft-ps-e60-moderate-ens3`); the judged budget is the last one the stop rule reached.
2. Comparator: `tools/results.py fe_C_twofast32_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt <ens3 student>_mmt`
   (control seed 1 0.701 at 5.33x; seed 2 0.694). The eval dirs live under different teachers' data dirs: pass
   paths (`05_distill/data/v4-ft-ps-e60-moderate/eval/<name>`, `05_distill/data/v4-ft-ps-e60-moderate-ens3/eval/<name>`)
   if the bare names don't resolve. Launch it through launch_job (it can run past 2 min).
   Frontier position by headline only (the key differs: new teacher).
3. LOOP.md steps 4-5: Results/Conclusion in notes.md, delete D1 from main's IDEAS.md (or trim it to what is
   untested), `tools/finish_experiment.sh d1-ens-teacher --arm distill --runs "<the ens3 rows>"`.
   Remove this HANDOFF.md first (finish_experiment refuses otherwise).

**If it died:** read the log's tail. Relaunch (resumes every stage, training from its checkpoint):
`cd .local/worktrees/d1-ens-teacher && tools/launch_job.sh distill.log -- bash chain.sh`
Disk: the packed shards go to main's `05_distill/data/v4-ft-ps-e60-moderate-ens3/shards` (C__twofast32 ~34 GB,
V ~4 GB, maybe plain C/V for yamnet mel if main.py packs them; / had 108 GB free).
