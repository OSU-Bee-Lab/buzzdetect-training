#!/usr/bin/env bash
# batch 38: llrd-depth4 medium CV alone (lite has no depth4 embeddings, so no lite smoke).
set -uo pipefail
PY=$(bash /home/luke/projects/buzzdetect-training/tools/python_path.sh)
export MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1
cd /home/luke/projects/buzzdetect-training/.local/worktrees/llrd-depth4 || exit 1
env TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_STREAM=1 TRUNK_BATCH=256 TRUNK_ACCUM=4 SCORE_CHUNK=128 TRUNK_LLRD=0.7 \
  "$PY" -u 03_train/main.py --name llrd-depth4 --set medium --embedder yamnet_trunk_pitchshift_depth4 \
  --translation general --epochs 30 -y --verbose 2>&1 | tee train.log | grep --line-buffered -vE "━|step - "
exit ${PIPESTATUS[0]}
