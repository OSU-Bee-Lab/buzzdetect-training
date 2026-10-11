# HANDOFF: fast32-d8-repeat

Batch 29 (4 experiments; **3 logged as of 21:40 on 2026-10-10: `ps-depth8-redraw` (0.497), `ps-d8-taps` (0.481) and `ps-d8-foldbal` (0.456)**; only `fast32-d8-repeat` remains). The chain (pid 2780967) is on its last step: foldbal exit 0 at 21:23, the distillation's trunk started training at 21:25 (63350 train steps in 8 stages, ~5 h expected, so ~02:30 on 10-11; no `[chain] left` time estimate had printed yet). In running order: `ps-depth8-redraw`, `ps-d8-taps`, `ps-d8-foldbal` (training arm), `fast32-d8-repeat` (distillation arm). Each has its own HANDOFF.md; an experiment is done when its slug is in `03_train/log.jsonl` (or, for the distill one, its students carry `"exp"` in `05_distill/log.jsonl`).

Jobs (both started by `tools/launch_job.sh`; adopt each with `tools/watch_job.sh --adopt <pid>`, then arm the one `tools/watch_job.sh` Monitor):
- pid 2744321, log `.local/worktrees/ps-depth8-redraw/train.log`: the control CV `ps-depth8-r3` (8 folds, ~16 min each, started 14:57).
- pid 2780967, log `.local/worktrees/ps-d8-taps/chain.log`: one chain that waits for 2744321 to exit, then runs the `ps-d8-taps` CV, the `ps-d8-foldbal` CV and the `fast32-d8-repeat` distillation, in that order, whatever each exits with. It prints `[chain] <step> exit N` after each step. Expected: taps ends ~19:20, foldbal ~21:30, distill ~5 h after that.

If the job this handoff needs is still running: log whatever experiments have already finished (their own handoffs), report progress, and park again (`loop_signal.sh park <minutes>`), keeping the handoffs of the unfinished ones. Delete a HANDOFF.md (git rm, commit) before `finish_experiment.sh` on its slug.

All four use env `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024` and args `--set medium --embedder yamnet_trunk_pitchshift_depth8 --translation general --epochs 30 -y` where a training command is given below; reruns resume at the first unfinished fold. One GPU job at a time.

## This experiment
Job: the chain, pid 2780967, step 3 (`[chain] distill exit N` in `.local/worktrees/ps-d8-taps/chain.log`); its `[chain] left: ...` lines (the distill chain's own, from 05_distill/main.py) give the ETA once it trains. Students: `fe_C_fast32_a0.50_d8_s2_c-buzz-rain-human_wsd{7000,14000,28000,56000}`.

When it finishes: `$(bash tools/python_path.sh) 05_distill/ladder_record.py frontier`, `ladder_record.py wsd --name fe_C_fast32_a0.50_d8_s2_c-buzz-rain-human_wsd`, and `tools/results.py fe_C_fast32_a0.50_d8_s1_c-buzz-rain-human_wsd56000 fe_C_fast32_a0.50_d8_s2_c-buzz-rain-human_wsd56000` through `tools/launch_job.sh` (it can run past 2 min). Write notes.md like `.local/worktrees/fast32h16-d8-repeat/notes.md` (same kind of experiment), then from main `tools/finish_experiment.sh fast32-d8-repeat --arm distill --summary "..." --runs "<the four seed-2 student names>"`.

If it died (or the loop killed the chain): from this worktree, `tools/launch_job.sh distill.log -- 05_distill/main.py --rung C --seed 2 --wsd 7000,14000,28000,56000 --runs "fast32:a0.50_d8:classes=ins_buzz+ambient_rain+human"` (resumes at stage and checkpoint level).
