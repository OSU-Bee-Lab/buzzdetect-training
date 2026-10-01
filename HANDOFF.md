# HANDOFF (batch 20, experiment agent)
Batch 20 = 4 experiments. Logged: ps-depth10, ps-cutout (2 of 4). In flight: ps-bntrain-depth10 (3 of 4). Remaining after it: 1 more (self-chosen: ps-noise, Gaussian noise on layer-11 map via TRUNK_NOISE, sigma ~0.1 of per-batch std; or another structural lever not in log.jsonl).

## ps-bntrain-depth10
- launch_job pid 327449, log `.local/worktrees/ps-bntrain-depth10/train.log`. Started 11:53; ~15 min/fold, ETA ~13:55. If still running: `tools/watch_job.sh --adopt 327449`, arm the one Monitor, report progress, stop (park again if >1.5h left).
- When finished: from main, `source tools/python_path.sh; W=$PWD/.local/worktrees/ps-bntrain-depth10; $PY tools/results.py $PWD/models/v4-ft-ps $W/models/ps-bntrain-depth10` (also compare to ps-depth10 0.473). Append Results/interpretation/Conclusion to `$W/notes.md`, `rm HANDOFF.md`, then `tools/finish_experiment.sh ps-bntrain-depth10 --summary ... --model $W/models/ps-bntrain-depth10 --baseline-model $PWD/models/v4-ft-ps --hypothesis ... --trust clean --conclusion ...`.
- If it died: from the worktree, relaunch (resumes folds):
  `export BUZZDETECT_CHUNK_FRAMES=48 TRUNK_LR_BACKBONE=1e-5 TRUNK_LR_HEAD=2e-4 TRUNK_BATCH=1024 TRUNK_FP16=1 TRUNK_BN_TRAIN=1; tools/launch_job.sh train.log -- 03_train/main.py --name ps-bntrain-depth10 --set medium --embedder yamnet_trunk_pitchshift_depth10 --translation general --epochs 30 -y`
