#!/usr/bin/env bash
# Shipped models for the moderate CVs (Luke, 2026-09-27). Waits for the CV queue
# (tools/queue_v4_moderate.sh, launch_job pid given as $1) to exit, then trains
# each shipped model with --skip-cv: every rotate + train fold pooled, epoch
# count read off that model's own rotation val_loss curves (train._consensus_epoch).
#
#   tools/launch_job.sh queue_v4_moderate_ship.log -- tools/queue_v4_moderate_ship.sh <cv pid>
#
# Ships a model only if its CV has all 5 folds on disk; a partial CV would take
# its epoch count from fewer curves. ps first: it has an ONNX export path; vu
# (vocoder) has none yet, so its shipped model is Keras-only.
cd "$(dirname "$(realpath "$0")")/.."
source tools/python_path.sh

export BUZZDETECT_CHUNK_FRAMES=48 TRUNK_FP16=1 TRUNK_LR_HEAD=2e-4 TRUNK_STREAM=1
FAILED=""
say() { echo "[ship] $(date '+%m-%d %H:%M') $*"; }
retry() {
  for i in 1 2 3 4 5 6; do
    "$@" && return 0
    say "retry $i: $* exited nonzero; resuming"
    sleep 5
  done
  return 1
}

if [ -n "$1" ]; then
  say "waiting for CV queue pid $1"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
  say "CV queue exited"
fi

ship() { # name embedder
  local n=$1 e=$2
  local folds; folds=$(find "models/$n/folds" -name summary.json 2>/dev/null | wc -l)
  if [ "$folds" -lt 5 ]; then
    say "skip $n: $folds/5 CV folds on disk"; FAILED="$FAILED ship:$n"; return
  fi
  say "ship $n ($e)"
  retry env TRUNK_LR_BACKBONE=1e-5 TRUNK_BATCH=1024 "$PY" -u 03_train/main.py --name "$n" \
    --set moderate --embedder "$e" --translation general --epochs 60 --skip-cv -y --verbose \
    || FAILED="$FAILED ship:$n"
}

ship v4-ft-ps-e60-moderate yamnet_trunk_pitchshift_depth12
ship v4-ft-vu-e60-moderate yamnet_trunk_vocoder_up_depth12

say "SHIP-DONE failed:${FAILED:- none}"
[ -z "$FAILED" ]
