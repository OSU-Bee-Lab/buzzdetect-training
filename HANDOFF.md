# HANDOFF: site-adv-w01 (batch 27, parked 2026-10-09 21:30)

Batch 27 status: **0 of 4 experiments logged.** Three are in flight as one dose
ladder (IDEAS.md item 27, reversal weight 0.1 / 0.3 / 1.0): `site-adv-w01`,
`site-adv-w03`, `site-adv-w10`, each with its own worktree and HANDOFF.md. The
4th is not started; pick it after reading the ladder (see "Fourth experiment").

## This worktree's part
`models/site-adv-w01`: 8-fold CV on medium, `TRUNK_ADV=0.1`, otherwise the
`ps-depth8` recipe. notes.md has the hypothesis and the code change.

## The job
pid **1675201**, log `.local/worktrees/site-adv-w01/train.log` (launch_job; this run alone).

Run `tools/watch_job.sh --adopt <pid>` for each live pid, then arm the one
`tools/watch_job.sh` Monitor. Measured rate: ~16 min per fold, ~2 h 10 min per
run. Order: w01 (started 20:58), then w03, then w10; all three done about 03:30.

**If it's still running, report progress and stop:** park again if more than
1.5 h remain (`loop_signal.sh park <minutes>`), otherwise re-arm and wait.

## When it finishes
`models/site-adv-w01/folds_sx.csv` exists and its log ends `exit 0`. Then, from
this worktree:

    python tools/results.py ps-depth8 site-adv-w01
    python tools/results.py ps-depth8-repeat site-adv-w01

Controls are the two no-adversary draws of the identical config (0.515, 0.509);
read the delta against both, and read the ladder as a dose response across the
three runs, not three separate verdicts. Also report the adversary's accuracy:
each `models/site-adv-w01/folds/<fold>/summary.json` has `site_acc_curve`
(chance is 1/`n_sites`); an adversary still naming the site well at epoch 30
means the tail kept site identity at this weight. Read docs/judging-results.md,
fill in notes.md (LOOP.md step 5), delete this HANDOFF.md and commit, then from
main:

    tools/finish_experiment.sh site-adv-w01 --summary "..." \
      --model .local/worktrees/site-adv-w01/models/site-adv-w01 --baseline-model ps-depth8-repeat \
      --hypothesis "..." --trust <trust> --conclusion "..."

Delete IDEAS.md item 27 from main's IDEAS.md with the last of the three
(`--commit-also IDEAS.md`), trimming it to any untested remainder.

## If it died
Stage 3 resumes on rerun (finished folds are skipped). From this worktree:

    TRUNK_ADV=0.1 TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train.log -- 03_train/main.py --name site-adv-w01 --set medium --embedder yamnet_trunk_pitchshift_depth8 --translation general --epochs 30 -y

One GPU job at a time: if the chain (pid 1677590) is still alive, let it finish
or relaunch only what it will not reach. An OOM in a train step would be the
adversary's extra memory on top of depth 8 at batch 1024: rerun with
`TRUNK_BATCH=512 TRUNK_ACCUM=2` (the same update; 03_train/CLAUDE.md).

## Fourth experiment
Not started. Suggested by the hypothesis's stated risk: if the ladder is flat or
falls with dose, test the conditional form (adversary on non-buzz frames only,
so site/label dependence in the pool can't be what it fights) at the best dose;
if a dose gains more than ~0.027 on the headline or lifts the hard folds, repeat
that dose instead.
