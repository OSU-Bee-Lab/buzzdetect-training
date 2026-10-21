# HANDOFF: gain-aug-d8 (batch 40, parked 2026-10-20 ~21:20)

Batch 40 runs four augmentation experiments on the depth8 recipe (yamnet_trunk_pitchshift_depth8,
TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024, --epochs 30, BUZZDETECT_CHUNK_FRAMES=48),
all against the matched control **ps-depth8-r3** (0.497; depth8 three-draw mean 0.507).
None of the four is logged yet: eq-aug-d8, snr-aug-d8, gain-aug-d8, pitch-aug-d8.
Each chain step: augment.py --all-folds (cached files skipped), a 1-epoch medium smoke, then the 8-fold CV
(~20 min/fold with the augmented pool).

This experiment's job is chain 2: launch_job pid 733264, log .local/worktrees/gain-aug-d8/chain2_b40.log (script gain-aug-d8/chain2_b40.sh; waits for chain 1's pid, then runs gain-aug-d8 then pitch-aug-d8).

- Resume watching: `tools/watch_job.sh --adopt <pid>`, then ONE `tools/watch_job.sh` Monitor.
- **If it's still running, report progress and stop** (park again if > 1.5 h remain).
- **When gain-aug-d8's CV finishes** (models/gain-aug-d8/folds_sx.csv in this worktree): check each fold's buzz logit SD in
  train.log, then `$(bash tools/python_path.sh) tools/results.py ps-depth8-r3 gain-aug-d8`, write Results and Conclusion
  in notes.md (Hypothesis is written), `git rm HANDOFF.md`, then from main:
  `tools/finish_experiment.sh gain-aug-d8 --summary ... --model .local/worktrees/gain-aug-d8/models/gain-aug-d8 --baseline-model ps-depth8-r3 --hypothesis ... --trust ... --conclusion ...`
  (eq-aug-d8 used IDEAS #32: delete that section from main's IDEAS.md and pass --commit-also IDEAS.md).
  Read eq/snr/gain/pitch against each other too.
- **If it died:** read the chain log tail. Relaunch (resumes: augment skips cached files, stage 3 skips finished folds):
  `cd .local/worktrees/gain-aug-d8 && tools/launch_job.sh chain2_b40.log -- bash chain2_b40.sh   (if chain 1 is also dead and unfinished, relaunch chain 1 first; chain 2's wait loop on pid 726415 then passes immediately, so launch chain 2 only after chain 1 has finished)`
