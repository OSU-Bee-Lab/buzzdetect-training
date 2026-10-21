# HANDOFF: snr-aug-d8 (batch 40, parked 2026-10-20 ~21:20)

Batch 40 runs four augmentation experiments on the depth8 recipe (yamnet_trunk_pitchshift_depth8,
TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024, --epochs 30, BUZZDETECT_CHUNK_FRAMES=48),
all against the matched control **ps-depth8-r3** (0.497; depth8 three-draw mean 0.507).
eq-aug-d8 is LOGGED (2026-10-21: 0.489, -0.008 +- 0.015 vs ps-depth8-r3, falsified). Remaining: snr-aug-d8, gain-aug-d8, pitch-aug-d8. Re-parked 2026-10-21 00:10, ETA ~9.5 h for all three.
Each chain step: augment.py --all-folds (cached files skipped), a 1-epoch medium smoke, then the 8-fold CV
(~20 min/fold with the augmented pool).

This experiment's job is chain 1: launch_job pid 726415, log .local/worktrees/eq-aug-d8/chain_b40.log (script eq-aug-d8/chain_b40.sh; runs eq-aug-d8 then snr-aug-d8).

- Resume watching: `tools/watch_job.sh --adopt <pid>`, then ONE `tools/watch_job.sh` Monitor.
- **If it's still running, report progress and stop** (park again if > 1.5 h remain).
- **When snr-aug-d8's CV finishes** (models/snr-aug-d8/folds_sx.csv in this worktree): check each fold's buzz logit SD in
  train.log, then `$(bash tools/python_path.sh) tools/results.py ps-depth8-r3 snr-aug-d8`, write Results and Conclusion
  in notes.md (Hypothesis is written), `git rm HANDOFF.md`, then from main:
  `tools/finish_experiment.sh snr-aug-d8 --summary ... --model .local/worktrees/snr-aug-d8/models/snr-aug-d8 --baseline-model ps-depth8-r3 --hypothesis ... --trust ... --conclusion ...`
  (eq-aug-d8 used IDEAS #32: delete that section from main's IDEAS.md and pass --commit-also IDEAS.md).
  Read eq/snr/gain/pitch against each other too.
- **If it died:** read the chain log tail. Relaunch (resumes: augment skips cached files, stage 3 skips finished folds):
  `cd .local/worktrees/eq-aug-d8 && tools/launch_job.sh chain_b40.log -- bash chain_b40.sh`
