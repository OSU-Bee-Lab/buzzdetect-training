#!/usr/bin/env bash
# Follow-up to queue_v4_grid.sh (2026-09-26). Fire once through launch_job:
#
#   tools/launch_job.sh queue_v4_psud.log -- tools/queue_v4_psud.sh
#
# 1. v4-ft-psud: grid winner v4-ft-ps plus an octave-DOWN third view
#    (yamnet_trunk_pitchshift_updown_depth12), 30 epochs, matched to v4-ft-ps.
# 2. v4-ft-ps-e60: v4-ft-ps rerun at 60 epochs, the epoch confirmation HANDOFF
#    asked for before the winner goes to moderate.
#
# Idempotent and retrying, same as queue_v4_grid.sh. Grep '^\[queue\]'.
cd "$(dirname "$(realpath "$0")")/.."
source tools/python_path.sh

export BUZZDETECT_CHUNK_FRAMES=48 TRUNK_FP16=1 TRUNK_LR_HEAD=2e-4
FAILED=""
say() { echo "[queue] $(date '+%m-%d %H:%M') $*"; }
retry() {
  for i in 1 2 3 4 5 6; do
    "$@" && return 0
    say "retry $i: $* exited nonzero; resuming"
    sleep 5
  done
  return 1
}

extract() { # set embedder
  say "extract $1 $2"
  retry "$PY" -u 02_set/main.py --set "$1" --embedder "$2" --workers 1 --verbose \
    || FAILED="$FAILED extract:$1/$2"
}

train() { # name embedder epochs [env assignments...]
  local n=$1 e=$2 ep=$3; shift 3
  say "train $n ($e, $ep epochs, $*)"
  retry env "$@" "$PY" -u 03_train/main.py --name "$n" --set medium --embedder "$e" \
    --translation general --epochs "$ep" -y --verbose \
    || FAILED="$FAILED train:$n"
}

FT="TRUNK_LR_BACKBONE=1e-5"

extract medium yamnet_trunk_pitchshift_updown_depth12
train v4-ft-psud   yamnet_trunk_pitchshift_updown_depth12 30 $FT TRUNK_BATCH=1024
train v4-ft-ps-e60 yamnet_trunk_pitchshift_depth12        60 $FT TRUNK_BATCH=1024

say "QUEUE-DONE failed:${FAILED:- none}"
[ -z "$FAILED" ]
