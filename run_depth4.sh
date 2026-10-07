#!/usr/bin/env bash
# Queued behind ps-depth6 (one GPU job at a time): wait for its launch_job pid,
# then train depth4. Rerunning this script resumes (finished folds are skipped).
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
wait_pid=${1:-}
if [ -n "$wait_pid" ]; then
  while kill -0 "$wait_pid" 2>/dev/null; do sleep 60; done
  echo "ps-depth6 job $wait_pid ended; starting ps-depth4"
fi
source /home/luke/projects/buzzdetect-training/tools/python_path.sh
export MALLOC_ARENA_MAX=2 TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_STREAM=1 \
       TRUNK_BATCH=256 TRUNK_ACCUM=4 SCORE_CHUNK=128
exec "$PY" -u 03_train/main.py --name ps-depth4 --set medium \
  --embedder yamnet_trunk_pitchshift_depth4 --translation general --epochs 30 -y --verbose
