# HANDOFF (batch 20, experiment agent)
Batch 20 = 4 experiments. Logged: ps-depth10 (1 of 4; +0.021 +/- 0.013 vs v4-ft-ps 0.452). In flight: ps-cutout (2 of 4). Remaining after it: 2 more, ideas below.

## ps-cutout
- launch_job pid 228792, log `.local/worktrees/ps-cutout/train.log`. Started 10:47; ~8-10 min/fold, ETA ~11:55. If still running: `tools/watch_job.sh --adopt 228792`, arm the one Monitor, report progress, stop (park again if >1.5h left).
- When finished: from main, `W=$PWD/.local/worktrees/ps-cutout; $PY tools/results.py $PWD/models/v4-ft-ps $W/models/ps-cutout` (absolute paths: friction already reported; control resolves from main's models/). Append Results + interpretation + Conclusion to `$W/notes.md` (Hypothesis/Changes are written), `rm HANDOFF.md` in the worktree, then `tools/finish_experiment.sh ps-cutout --summary ... --model $W/models/ps-cutout --baseline-model $PWD/models/v4-ft-ps --hypothesis ... --trust clean --conclusion ...` (trust choices: clean|caveated|artifact).
- If it died: relaunch (resumes finished folds), from the worktree:
  `export BUZZDETECT_CHUNK_FRAMES=48 TRUNK_LR_BACKBONE=1e-5 TRUNK_LR_HEAD=2e-4 TRUNK_BATCH=1024 TRUNK_FP16=1 TRUNK_CUTOUT=0.5; tools/launch_job.sh train.log -- 03_train/main.py --name ps-cutout --set medium --embedder yamnet_trunk_pitchshift_depth12 --translation general --epochs 30 -y`

## Experiments 3 and 4 (IDEAS queue is empty; self-chosen, matched control v4-ft-ps 0.452, config above)
Worktree recipe: `bash tools/setup_worktree.sh <slug>`; to add embedder knobs, `rm embedders/yamnet_trunk_depth12 && cp -r <main>/embedders/yamnet_trunk_depth12 embedders/` (see ps-cutout's copy for the TRUNK_* env-knob pattern).
- 3: ps-depth10-bntrain? No: better `ps-depth10-repeat` is weak value. Suggested: ps-noise (Gaussian noise on the layer-11 map, TRUNK_NOISE, relative sigma ~0.1 of per-batch std) or ps-viewdrop; pick one structural lever not already in log.jsonl.
- 4: ps-bntrain-depth10 (TRUNK_BN_TRAIN=1 knob exists in `.local/worktrees/ps-bntrain-gmp/embedders/yamnet_trunk_depth12/embedder.py`; run with embedder yamnet_trunk_pitchshift_depth10, extraction already cached; control v4-ft-ps and compare to ps-depth10 0.473 too).
