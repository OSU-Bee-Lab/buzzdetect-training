# HANDOFF: twofast32-d8 (batch 31, experiment 3 of 4)

**Batch state:** 2 of 4 experiments are logged (`ps-d8-ensemble` in `03_train/log.jsonl`, `matmul-stft` in
`05_distill/log.jsonl`). This one is the third. A fourth is still owed after it (suggestion at the bottom).

## The job
- pid **4177608** (launch_job's own pid), log `.local/worktrees/twofast32-d8/distill.log`, launched
  2026-10-11 11:48 from this worktree: `tools/launch_job.sh distill.log -- bash chain.sh`.
- `chain.sh` runs `05_distill/main.py --rung C --runs "twofast32:a0.50_d8:<buzz+rain+human>
  twofast32:a0.25_d8:<buzz+rain+human>" --wsd-max 56000 --wsd-stop 1.3`, then re-exports every judged
  branch with the pad-free front end (`reexport.py --fe-impl mmt --suffix _mmt`).
- At 12:13 the chain printed `left: at most 120750 train steps in 15 train stages, about 8h14m`
  (twofast32 trains ~4 steps/s). Expect it to end around 20:30, later if the stop rule lets both run to 56000.

Adopt it: `tools/watch_job.sh --adopt 4177608`, then the one `tools/watch_job.sh` Monitor.

## If it is still running
Report progress (the Monitor's state line carries `[chain] left: ...`) and stop: park again if more than
1.5 h remain, otherwise keep the Monitor armed until `[launch_job] exit 0`.

## When it finishes (exit 0)
1. Students: `fe_C_twofast32_a0.50_d8_s1_c-buzz-rain-human_wsd{7000,14000,28000,56000}` and the
   `a0.25_d8` four (fewer if `--wsd-stop` ended a trunk), each with a `_mmt` twin. Headlines are the
   same in both; **speed comes from the `_mmt` rows** (the plain rows still carry the waveform Pad).
2. Comparator (matched control): `fe_C_twofast32_a0.50_s1_c-buzz-rain-human_wsd56000` (0.691; 4.81x as
   `..._mmt`). Read:
   - `$(bash tools/python_path.sh) 05_distill/ladder_record.py frontier`
   - `ladder_record.py wsd --name fe_C_twofast32_a0.50_d8_s1_c-buzz-rain-human_wsd` (and the a0.25_d8 trunk)
   - `tools/launch_job.sh results.log -- $(bash tools/python_path.sh) tools/results.py
     fe_C_twofast32_a0.50_s1_c-buzz-rain-human_wsd56000 fe_C_twofast32_a0.50_d8_s1_c-buzz-rain-human_wsd56000`
     (then the same control against the a0.25_d8 student, and a0.25_d8 against
     `fe_C_fast32h16_a0.50_d8_s1_c-buzz-rain-human_wsd56000`).
3. The test, set before training (notes.md): `a0.50_d8` (bench 5.36x) is a new frontier point if its
   headline is above **0.676** (0.02 over fast32h16 a0.75_d8's 0.656 at 5.57x, the best at its speed or
   faster); `a0.25_d8` (bench 6.20x) if above **0.666** (fast32h16 a0.50_d8 0.646 / 0.642 at 6.7x). Use the
   trained students' own `_mmt` speeds, not the bench's. A gain over 0.02 owes one seed-2 repeat before
   it counts (`ladder_record.py repeats` prints the command; export it with `mmt` too): say so in the
   conclusion and queue it in IDEAS.md rather than running it inside this experiment.
4. Steps 4-5 of LOOP.md: finish `notes.md` (Changes, Results, Conclusion: headline and speed together,
   folds against their SDs, tiers), then from main:
   `tools/finish_experiment.sh twofast32-d8 --arm distill --summary "..." --runs "<all 16 names, plain and _mmt>"`.
   Delete this HANDOFF.md first (`git rm`), finish_experiment refuses a worktree that still has one.

## If it died
Read the tail of `distill.log`. Every stage resumes, so relaunch the same command from this worktree:

    cd .local/worktrees/twofast32-d8 && tools/launch_job.sh distill.log -- bash chain.sh

A GPU out-of-memory in a val/flip pass is an unbatched call (CLAUDE.md), not a reason for `--cpu`. If
only the re-export part failed, the same relaunch skips the finished training and redoes it.

## Experiment 4 (not started)
Suggested, in order of preference:
1. **IDEAS 30(a)**, training arm: two more plain `ps-depth8` draws on `medium` (~2 h 10 min each on the GPU,
   recipe in `.local/worktrees/ps-depth8-redraw/notes.md`: `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1
   TRUNK_BATCH=1024`, `--embedder yamnet_trunk_pitchshift_depth8 --epochs 30`), then score the 4- and
   5-draw same-recipe means with `tools_exp/ensemble_predictions.py` from `exp/ps-d8-ensemble` against
   `ps-d8-ens3` (0.558) and the mixed-depth five (0.575). It needs the GPU, so it starts after this chain.
2. **IDEAS D2**, distillation arm: the 64-band hop-16 front end (needs a `cache_fe` pass and 25-45 GB).
