# HANDOFF: site-conf-w01 (batch 27, parked 2026-10-10 04:25)

Batch 27 status: **3 of 4 experiments logged** (`site-adv-w01`, `site-adv-w03`,
`site-adv-w10`, all `artifact`: gradient reversal diverged training). This is
the 4th and last; nothing else remains after it.

## This worktree's part
`models/site-conf-w01`: 8-fold CV on medium, `TRUNK_ADV=0.1 TRUNK_ADV_MODE=confuse`
(bounded site term: KL to uniform instead of gradient reversal), otherwise the
`ps-depth8` recipe. notes.md has the hypothesis and the code change; Results and
Conclusion are still to write.

## The job
pid **1911772**, log `.local/worktrees/site-conf-w01/train.log` (launch_job; this run alone).
Started 03:52; ~16 min per fold (1 fold done and the 2nd at epoch 27/30 at
04:22), so about 06:00.

Run `tools/watch_job.sh --adopt 1911772`, then arm the one `tools/watch_job.sh` Monitor.

**If it's still running, report progress and stop:** park again if more than
1.5 h remain (`loop_signal.sh park <minutes>`), otherwise re-arm and wait.

## When it finishes
`models/site-conf-w01/folds_sx.csv` exists and the log ends `exit 0`. From this worktree:

    python tools/results.py ps-depth8-repeat site-conf-w01
    python tools/results.py ps-depth8 site-conf-w01

Controls are the two no-adversary draws (0.515, 0.509). Before reading the
delta, check the precondition in notes.md's Hypothesis from each
`models/site-conf-w01/folds/**/summary.json`: `val_loss_curve` must stay in the
0.6-0.9 basin (a spike means it diverged like the ladder and the result is an
artifact), and `site_acc_curve` says whether the adversary was pushed toward
chance (1/48 = 0.021). Stable but `site_acc` still far above chance is "no test
at this dose", not a verdict on site invariance. Read docs/judging-results.md,
fill in notes.md's Results and Conclusion (LOOP.md step 5; the three
`site-adv-*` notes show the curve table), delete this HANDOFF.md and commit,
then from main:

    tools/finish_experiment.sh site-conf-w01 --summary "..." \
      --model .local/worktrees/site-conf-w01/models/site-conf-w01 --baseline-model ps-depth8-repeat \
      --hypothesis "..." --trust <trust> --conclusion "..."

IDEAS.md item 27 is already deleted. If this run leaves a live follow-up
(a higher weight, or the conditional form on non-buzz frames only), add it to
main's IDEAS.md and pass `--commit-also IDEAS.md`. Then
`loop_signal.sh done "site-adv-w01 site-adv-w03 site-adv-w10 site-conf-w01"`.

## If it died
Stage 3 resumes on rerun (finished folds are skipped). From this worktree:

    TRUNK_ADV=0.1 TRUNK_ADV_MODE=confuse TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train.log -- 03_train/main.py --name site-conf-w01 --set medium --embedder yamnet_trunk_pitchshift_depth8 --translation general --epochs 30 -y

On an OOM in a train step: `TRUNK_BATCH=512 TRUNK_ACCUM=2` (the same update; 03_train/CLAUDE.md).
