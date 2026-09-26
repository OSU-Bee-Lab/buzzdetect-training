#!/usr/bin/env bash
# The pitch-shift method x direction grid (2026-09-26, HANDOFF.md). Fire once
# through launch_job:
#
#   tools/launch_job.sh queue_v4_psd.log -- tools/queue_v4_psd.sh
#
# Six CVs, all 30 epochs and otherwise matched to v4-ft (no shift, 0.375),
# v4-ft-ps (resample up, 0.452) and v4-ft-psud (resample up + centre down, 0.452):
#
#   v4-ft-psd    resample down (centre half)       yamnet_trunk_pitchshift_down_depth12
#   v4-ft-pshd   resample down, both halves        yamnet_trunk_pitchshift_halves_down_depth12
#   v4-ft-vd     vocoder down                      yamnet_trunk_vocoder_down_depth12
#   v4-ft-vu     vocoder up                        yamnet_trunk_vocoder_up_depth12
#   v4-ft-vud    vocoder up + down                 yamnet_trunk_vocoder_updown_depth12
#   v4-ft-pshud  resample up + down both halves    yamnet_trunk_pitchshift_halves_updown_depth12
#
# Luke asked for all six at once (out of office) rather than gating the later
# ones on psd. pshud is last: four views is the largest host-RAM load, and if it
# gets killed the other five are already done.
#
# Idempotent and retrying, same as queue_v4_psud.sh. Grep '^\[queue\]'.
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

run() { # name embedder
  extract medium "$2"
  train "$1" "$2" 30 $FT TRUNK_BATCH=1024
}

run v4-ft-psd   yamnet_trunk_pitchshift_down_depth12
run v4-ft-pshd  yamnet_trunk_pitchshift_halves_down_depth12
run v4-ft-vd    yamnet_trunk_vocoder_down_depth12
run v4-ft-vu    yamnet_trunk_vocoder_up_depth12
run v4-ft-vud   yamnet_trunk_vocoder_updown_depth12
run v4-ft-pshud yamnet_trunk_pitchshift_halves_updown_depth12

say "QUEUE-DONE failed:${FAILED:- none}"
[ -z "$FAILED" ]
