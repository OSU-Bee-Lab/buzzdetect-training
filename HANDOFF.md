# HANDOFF: pseudo-label (batch 28, parked 2026-10-10 07:40)

Batch 28 status: **2 of 4 experiments logged** (`site-neg-w01`, `site-ent-ramp`,
both gated after one fold, `artifact`; the site-invariance family is now under
IDEAS.md "Ruled out"). This is the 3rd. The 4th is not started; see the end.

## This worktree's part
`models/pseudo-label` (the student, the reported model) and
`models/pseudo-label_teacher` (each rotation's teacher scored on the same
held-out fold: the matched control). IDEAS.md item 28, two-stage self-training
on `ps-depth8`; notes.md has the hypothesis and the changes, Results and
Conclusion are still to write. The unlabelled set is already extracted
(`02_set/sets/medium_unlabeled/embeddings`, in this worktree only, ~1 GB).

First rotation's teacher line (07:25): `1_29` teacher 0.441, logit SD 1.14;
pseudo-labels from 41 folds: 349 positive, 5956 negative of 20992 frames (70%
fell in the dropped band).

## The job
pid **2225316**, log `.local/worktrees/pseudo-label/train.log` (launch_job; this run alone).
Started 07:09; a rotation is teacher (16 min) + student (~17 min), so 8
rotations end about **11:40**.

Run `tools/watch_job.sh --adopt 2225316`, then arm the one `tools/watch_job.sh` Monitor.

**If it's still running, report progress and stop:** park again if more than
1.5 h remain (`loop_signal.sh park <minutes>`), otherwise re-arm and wait.

## When it finishes
The log ends `exit 0` and `models/pseudo-label/folds_sx.csv` exists. The teacher
has per-fold predictions but no table yet. From this worktree:

    $(bash tools/python_path.sh) 03_train/resummarize.py pseudo-label_teacher
    $(bash tools/python_path.sh) tools/results.py pseudo-label_teacher pseudo-label   # the matched pair
    $(bash tools/python_path.sh) tools/results.py ps-depth8-repeat pseudo-label_teacher  # where the teacher draw sits
    $(bash tools/python_path.sh) tools/results.py ps-depth8-repeat pseudo-label

Check every fold line's `buzz logit SD` (teacher and student) in train.log, and
each student `summary.json`'s `pseudo` counts. Read docs/judging-results.md;
the student-vs-teacher delta is the result (same run, same pool, same held-out
folds; its SD is eval sampling plus ~0.007 training noise on the headline).
A headline gain over ~0.027 gets one repeat run before it counts. Fill in
notes.md's Results and Conclusion (LOOP.md step 5), `rm HANDOFF.md`, commit,
delete item 28's section from **main's** IDEAS.md, then from main:

    tools/finish_experiment.sh pseudo-label --summary "..." \
      --model .local/worktrees/pseudo-label/models/pseudo-label \
      --baseline-model .local/worktrees/pseudo-label/models/pseudo-label_teacher \
      --hypothesis "..." --trust <trust> --conclusion "..." --commit-also IDEAS.md

## If it died
Stage 3 resumes on rerun: finished student folds are skipped, and an unfinished
rotation retrains its teacher and student. From this worktree:

    PSEUDO_SET=medium_unlabeled TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train.log -- 03_train/main.py --name pseudo-label --set medium --embedder yamnet_trunk_pitchshift_depth8 --translation general --epochs 30 -y

(`train.log` is overwritten by a relaunch; copy it first if the teacher lines of
finished rotations are needed. They are also recoverable from
`models/pseudo-label_teacher/folds/*/predictions.csv`.)

## The 4th experiment (not started, no worktree)
Planned: the distillation arm's pending repeat, as `ladder_record.py repeats`
prints it (fast32h16 a0.50_d8 gained +0.034 over everything at its speed or
faster on one seed):

    tools/launch_job.sh distill.log -- 05_distill/main.py --rung C --seed 2 --wsd 7000,14000,28000,56000 --runs "fast32h16:a0.50_d8:classes=ins_buzz+ambient_rain+human"

from a fresh `bash tools/setup_worktree.sh fast32h16-d8-repeat` worktree, after
this job has exited (one GPU job at a time). If pseudo-label shows a gain that
needs its repeat run, do that repeat instead: it outranks this.
Then `loop_signal.sh done "site-neg-w01 site-ent-ramp pseudo-label <4th slug>"`.
