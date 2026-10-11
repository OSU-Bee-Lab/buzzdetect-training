# HANDOFF: fast32h16-a025-repeat (batch 30, experiment 4 of 4)

Batch 30 status at park (2026-10-11 07:40): **3 of 4 logged** (`fused-blocks`, `conv-stft` in the distill
arm; `ps-d8-aves` in the training arm). This is the last one; nothing else remains after it.

## The job
- pid **3788001**, log `.local/worktrees/fast32h16-a025-repeat/distill.log`. Adopt it:
  `tools/watch_job.sh --adopt 3788001`, then one `tools/watch_job.sh` Monitor.
- Started 07:07. The chain's own estimate at 07:27: at most 57400 train steps in 7 stages, about 2h57m,
  so ETA ~10:25 (sooner if `--wsd-stop` ends the trunk early; this command passes explicit budgets, so
  expect all four).
- **If it is still running, report progress and stop** (re-arm; park again only if > 1.5 h remain by
  the state line's `left:` estimate).

## When it finishes (exit 0)
notes.md has the hypothesis and the command. No code changes.
1. `$(bash tools/python_path.sh) 05_distill/ladder_record.py frontier` from this worktree: the seed-2 rows
   are `fe_C_fast32h16_a0.25_s2_c-buzz-rain-human_wsd{7000,14000,28000,56000}`.
2. Comparator, through launch_job (it can run past 2 min; `--cpu` is fine):
   `tools/launch_job.sh --cpu results.log -- tools/results.py fe_C_fast32h16_a0.25_s1_c-buzz-rain-human_wsd56000 fe_C_fast32h16_a0.25_s2_c-buzz-rain-human_wsd56000`
3. notes.md Results/Conclusion, as `exp/fast32-d8-repeat`'s notes: seed 2 against seed 1 at 56000 (seed 1:
   0.612 at 2.75x), the budget curve for both seeds, speed, tiers. Then say what the two-seed mean does to
   `fused-blocks`' reading: `a0.25_d8_f2` was 0.630 at 2.73x, +0.018 +- 0.008 over seed 1.
4. Delete this file, commit, then from main:
   `tools/finish_experiment.sh fast32h16-a025-repeat --arm distill --summary "..." --runs "<the four s2 names>"`
5. `tools/loop_signal.sh done "fused-blocks ps-d8-aves conv-stft fast32h16-a025-repeat"`.

## If it died
Relaunch from this worktree (the chain resumes at stage and checkpoint level):
`tools/launch_job.sh distill.log -- 05_distill/main.py --rung C --seed 2 --wsd 7000,14000,28000,56000 --runs "fast32h16:a0.25:classes=ins_buzz+ambient_rain+human"`
