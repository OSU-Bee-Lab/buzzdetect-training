# HANDOFF: ens3-moderate (batch 32, experiment 4 of 4)

Batch state: **3 of 4 experiments logged** (`ps-d8-ktails`, `twofast32-d8-s2`, `yam64h16`). This is the
fourth and last: when it is logged, send `loop_signal.sh done "ps-d8-ktails twofast32-d8-s2 yam64h16 ens3-moderate"`.

## The job
- pid **2660110**, log `.local/worktrees/ens3-moderate/train.log`, launched 2026-10-13 07:22 with
  `tools/launch_job.sh train.log -- bash chain.sh` from this worktree.
- Adopt it: `tools/watch_job.sh --adopt 2660110`, then one `tools/watch_job.sh` Monitor.
- It trains two more draws of the teacher's recipe on `moderate` (Luke approved D1's `moderate` runs,
  2026-10-11): `v4-ft-ps-e60-moderate-r2`, then `-r3` (5 rotations each, 60 epochs, no shipped model), then
  writes the logit means `ens3-moderate`, `ens2ab/ac/bc-moderate` (seconds). Draw 1 took ~1 h 45-50 min per
  fold on 2026-09-27, so expect ~9 h per draw: r2 done about 16:30 on 10-13, r3 and the end about 01:30-02:00
  on 2026-10-14. The Monitor's state line counts folds done (10 in all).

## If it is still running
Report progress (folds done of 10, time per fold so far) and stop: re-park if more than 1.5 h remain.

## When it finishes (`[launch_job] exit 0`)
1. Check each fold line's `buzz logit SD` in train.log (0.65-1.2 is trained; ~0.1 is a collapse).
2. `tools/launch_job.sh results.log -- bash results.sh` from this worktree (ens3 against each member, and
   the new draws against draw 1). Headlines of the pairs: `models/ens2*-moderate/folds_sx.csv` total row.
3. Judge against notes.md (docs/judging-results.md first): prediction is the 3-draw mean more than +0.02 over
   the members' mean (draw 1 is 0.574), thresholds lower on every fold; falsifier is within 0.02. Report the
   three draws' spread, the fold deltas with ± SD, the tiers, and the per-fold thresholds
   (`folds_sx.csv` `threshold` column, members against ens3).
4. Write notes.md Results / Conclusion, delete this HANDOFF.md, commit. In main's IDEAS.md trim D1 to its
   untested remainder (the confirmation is done: say what it found in the Evidence line; the deploy +
   re-cache + distil part stays). Then from main:
   `tools/finish_experiment.sh ens3-moderate --summary "..." --model .local/worktrees/ens3-moderate/models/ens3-moderate
   --baseline-model v4-ft-ps-e60-moderate --hypothesis "..." --trust <trust> --conclusion "..." --commit-also IDEAS.md`
   (only if `git diff IDEAS.md` shows nothing but that edit). `git add -f` the ens model's `folds_sx.csv` if
   the finish script does not pick it up.

## If it died (exit non-zero, or pid gone with no exit 0)
Read the tail of train.log. Stage 3 resumes on rerun (finished folds are skipped, a half-trained fold is
wiped and retrained). A GPU OOM in scoring after a fold trained: set `SCORE_CHUNK` lower (see
`03_train/CLAUDE.md`), not `--cpu`. If `ensemble_predictions.py` refuses ("rows differ"), draw 1's
predictions predate a set change: record that, and score `ens2bc-moderate` (the two new draws) against its
members as the result instead. Relaunch from this worktree:
`tools/launch_job.sh train.log -- bash chain.sh`
