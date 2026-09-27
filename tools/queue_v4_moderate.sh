#!/usr/bin/env bash
# The era's top configurations on `moderate` (Luke's call, 2026-09-27). Fire once
# through launch_job:
#
#   tools/launch_job.sh queue_v4_moderate.log -- tools/queue_v4_moderate.sh
#
# Two CVs, 60 epochs (v4-ft-ps-e60's budget), otherwise matched to the medium grid:
#
#   v4-ft-ps-e60-moderate   resample up   yamnet_trunk_pitchshift_depth12   (medium e60: 0.468)
#   v4-ft-vu-e60-moderate   vocoder up    yamnet_trunk_vocoder_up_depth12   (medium e30: 0.455)
#
# v4-ft-pshud (0.455, four views) was dropped: twice the cost for a tie.
# A two-view trunk pool on moderate is ~18 GB of float16, past host RAM, so the
# training pool streams from disk (TRUNK_STREAM=1, train._StreamPool).
#
# Idempotent and retrying, same as queue_v4_psd.sh. Grep '^\[queue\]'.
cd "$(dirname "$(realpath "$0")")/.."
source tools/python_path.sh

export BUZZDETECT_CHUNK_FRAMES=48 TRUNK_FP16=1 TRUNK_LR_HEAD=2e-4 TRUNK_STREAM=1
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
  retry env "$@" "$PY" -u 03_train/main.py --name "$n" --set moderate --embedder "$e" \
    --translation general --epochs "$ep" -y --verbose \
    || FAILED="$FAILED train:$n"
}

run() { # name embedder
  extract moderate "$2"
  train "$1" "$2" 60 TRUNK_LR_BACKBONE=1e-5 TRUNK_BATCH=1024
}

run v4-ft-ps-e60-moderate yamnet_trunk_pitchshift_depth12
run v4-ft-vu-e60-moderate yamnet_trunk_vocoder_up_depth12

say "QUEUE-DONE failed:${FAILED:- none}"
[ -z "$FAILED" ]
