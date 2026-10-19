# HANDOFF: ps2-depth8 (batch 37, experiment 2 of 4; 1 of 4 logged: yam64h16sp2. Remaining after this: ps-depth8-e60 (worktree made, hypothesis committed, not launched) and one more, TBD)
Job: pid 3186636 (launch_job.sh, `bash chain.sh`), log `.local/worktrees/ps2-depth8/chain.log`.
Resume: `tools/watch_job.sh --adopt 3186636`, then one `tools/watch_job.sh` Monitor (30 min, re-arm).
If it's still running, report progress and stop (park again if > 1.5 h remain; folds take ~25 min each).
When it finishes (exit 0): check each fold's `buzz logit SD` in chain.log, then
`tools/results.py ps-depth8-r3 ps2-depth8` (via launch_job), also note it against the three-draw ps-depth8 mean (0.507).
Fill notes.md Results/Conclusion against the Hypothesis falsifier (LOOP.md step 4, docs/judging-results.md), then step 5:
`tools/finish_experiment.sh ps2-depth8 --summary "..." --model .local/worktrees/ps2-depth8/models/ps2-depth8 --baseline-model ps-depth8-r3 --hypothesis "..." --trust clean --conclusion "..."`.
A gain > ~0.03 over the three-draw mean earns one repeat (LOOP.md): that would be a good 4th experiment (`ps2-depth8-repeat`).
Then launch ps-depth8-e60 from its worktree:
`tools/launch_job.sh train.log -- env TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 $(bash tools/python_path.sh) 03_train/main.py --name ps-depth8-e60 --set medium --embedder yamnet_trunk_pitchshift_depth8 --translation general --epochs 60 -y` (~4.3 h: park).
If it died: read chain.log. Relaunch from this worktree with `tools/launch_job.sh chain.log -- bash chain.sh` (extraction is done and skips; 03_train resumes finished folds).
A GPU OOM in training: batch 1024 already OOMed (chain_oom1024.log); go to `TRUNK_BATCH=256 TRUNK_ACCUM=4 SCORE_CHUNK=128` in chain.sh and note it in notes.md.
