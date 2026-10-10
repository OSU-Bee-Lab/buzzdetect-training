# HANDOFF: fast32h16-d8-repeat (batch 28, parked 2026-10-10 12:13)

Batch 28 status: **3 of 4 experiments logged** (`site-neg-w01`, `site-ent-ramp`,
`pseudo-label`). This is the 4th and last; nothing else remains.

## This worktree's part
Distillation arm, the repeat rule: seed 2 of `fast32h16:a0.50_d8` at rung C, WSD
budgets 7000/14000/28000/56000. notes.md has the hypothesis; Results and
Conclusion are still to write. The students are
`fe_C_fast32h16_a0.50_d8_s2_c-buzz-rain-human_wsd<budget>`; the matched control
is the seed-1 row at the same budget (`..._s1_..._wsd56000`).

So far: wsd7000 scored 0.612 (incl. quiet 0.508) at 12:09.

## The job
pid **2558428**, log `.local/worktrees/fast32h16-d8-repeat/distill.log`
(launch_job; this run alone). Started 11:41. At 12:10:51 the chain printed
`left: at most 56350 train steps in 6 train stages, about 2h35m`, so it ends
about **14:45**.

Run `tools/watch_job.sh --adopt 2558428`, then arm the one `tools/watch_job.sh` Monitor.

**If it's still running, report progress and stop:** read the state line's
`left:` estimate; park again if more than 1.5 h remain
(`loop_signal.sh park <minutes>`), otherwise re-arm and wait.

## When it finishes
The log ends `exit 0` and the s2 rows are in main's `05_distill/log.jsonl`
(main.py's record stage writes them). From main:

    $(bash tools/python_path.sh) 05_distill/ladder_record.py frontier
    $(bash tools/python_path.sh) 05_distill/ladder_record.py wsd --name fe_C_fast32h16_a0.50_d8_s2_c-buzz-rain-human_wsd
    tools/launch_job.sh results.log -- tools/results.py \
      05_distill/data/v4-ft-ps-e60-moderate/eval/fe_C_fast32h16_a0.50_d8_s1_c-buzz-rain-human_wsd56000 \
      05_distill/data/v4-ft-ps-e60-moderate/eval/fe_C_fast32h16_a0.50_d8_s2_c-buzz-rain-human_wsd56000

(check the eval dir names with `ls` first; results.py on eval dirs can run past
2 min, hence launch_job.) Judge at the last budget recorded. The question is
whether seed 2 is within seed noise (~0.01-0.02) of seed 1 and still more than
0.02 above every rung-C student at its speed or faster; report headline and
speed (`x YAMNet`) together, and the tiers. Fill in notes.md's Results and
Conclusion, `rm HANDOFF.md`, commit, then from main:

    tools/finish_experiment.sh fast32h16-d8-repeat --arm distill --summary "..." \
      --runs "<the s2 student names, space-separated>"

Then `loop_signal.sh done "site-neg-w01 site-ent-ramp pseudo-label fast32h16-d8-repeat"`.

## If it died
main.py resumes at stage and training-checkpoint level on rerun. From this worktree:

    tools/launch_job.sh distill.log -- 05_distill/main.py --rung C --seed 2 --wsd 7000,14000,28000,56000 --runs "fast32h16:a0.50_d8:classes=ins_buzz+ambient_rain+human"

(`distill.log` is overwritten by a relaunch; copy it first.)
