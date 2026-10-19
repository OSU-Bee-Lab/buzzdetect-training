# HANDOFF: ps-depth8-e60 (batch 37, experiment 3 of 4; 2 of 4 logged: yam64h16sp2, ps2-depth8. Remaining after this: one more, TBD)
Job: pid 3390005 (launch_job.sh, 03_train/main.py), log `.local/worktrees/ps-depth8-e60/train.log`. Launched 15:21, ~32 min/fold x 8 folds, ETA ~19:40.
Resume: `tools/watch_job.sh --adopt 3390005`, then one `tools/watch_job.sh` Monitor (30 min, re-arm).
If it's still running, report progress and stop (park again if > 1.5 h remain).
When it finishes (exit 0): check each fold's `buzz logit SD` and best epoch in train.log (does best land before 60? that is the overfit question), then
`tools/results.py ps-depth8-r3 ps-depth8-e60` (via launch_job); note it against the three-draw ps-depth8 mean (0.507).
Fill notes.md Results/Conclusion against the Hypothesis falsifier (LOOP.md step 4), then step 5:
`tools/finish_experiment.sh ps-depth8-e60 --summary "..." --model .local/worktrees/ps-depth8-e60/models/ps-depth8-e60 --baseline-model ps-depth8-r3 --hypothesis "..." --trust clean --conclusion "..."`.
A gain > ~0.03 over the three-draw mean earns one repeat (LOOP.md); that would be the 4th experiment. Otherwise pick the 4th per LOOP.md (IDEAS.md training queue holds only #29 BEATs/AVES2: check it against the AVES injunction first; distillation is closed by injunction).
If it died: read train.log. Relaunch from this worktree with
`tools/launch_job.sh train.log -- env TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 $(bash tools/python_path.sh) 03_train/main.py --name ps-depth8-e60 --set medium --embedder yamnet_trunk_pitchshift_depth8 --translation general --epochs 60 -y` (finished folds skip).
A GPU OOM: go to `TRUNK_BATCH=512 TRUNK_ACCUM=2 SCORE_CHUNK=512` and note it in notes.md.
