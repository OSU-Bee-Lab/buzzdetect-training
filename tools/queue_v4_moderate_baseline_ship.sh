#!/usr/bin/env bash
# Shipped model for cv-baseline-v4-moderate (Luke, 2026-09-28). Waits for the CV
# queue (tools/queue_v4_moderate_baseline.sh, launch_job pid given as $1) to
# exit, then trains the shipped model with --skip-cv, matched to the CV's flags.
#
#   tools/launch_job.sh queue_v4_moderate_baseline_ship.log -- tools/queue_v4_moderate_baseline_ship.sh <cv pid>
#
# Ships only if the CV has all 5 folds on disk. Re-launch with no pid if the CV
# queue is gone.
cd "$(dirname "$(realpath "$0")")/.."
source tools/python_path.sh

export BUZZDETECT_CHUNK_FRAMES=48
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

n=cv-baseline-v4-moderate
folds=$(find "models/$n/folds" -name summary.json 2>/dev/null | wc -l)
if [ "$folds" -lt 5 ]; then
  say "SHIP-DONE failed: skip $n: $folds/5 CV folds on disk"; exit 1
fi
say "ship $n (yamnet)"
if retry "$PY" -u 03_train/main.py --name "$n" --set moderate --embedder yamnet \
    --translation general --epochs 400 --skip-cv -y --verbose; then
  say "SHIP-DONE failed: none"
else
  say "SHIP-DONE failed: ship:$n"; exit 1
fi
