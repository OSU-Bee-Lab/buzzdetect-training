#!/usr/bin/env bash
# ps-fast isolation: two CVs, each with a lite smoke first. Idempotent and
# retrying; grep '^\[queue\]'.
source /home/luke/projects/buzzdetect-training/tools/python_path.sh
export BUZZDETECT_CHUNK_FRAMES=48 TRUNK_FP16=1 TRUNK_LR_HEAD=2e-4 TRUNK_LR_BACKBONE=1e-5 TRUNK_BATCH=1024
WT=/home/luke/projects/buzzdetect-training/.local/worktrees
say() { echo "[queue] $(date '+%m-%d %H:%M') $*"; }
retry() {
  for i in 1 2 3; do
    "$@" && return 0
    say "retry $i: $* exited nonzero; resuming"
    sleep 5
  done
  say "FAILED: $*"; return 1
}
run() { # slug embedder name
  cd "$WT/$1" || return 1
  say "$1: smoke extract lite"
  retry "$PY" -u 02_set/main.py --set lite --embedder "$2" --workers 1 --verbose || return 1
  say "$1: smoke train lite"
  retry "$PY" -u 03_train/main.py --name "test_$1_smoke" --set lite --embedder "$2" \
    --translation general --epochs 2 -y --verbose || return 1
  say "$1: extract medium"
  retry "$PY" -u 02_set/main.py --set medium --embedder "$2" --workers 1 --verbose || return 1
  say "$1: train $3"
  retry "$PY" -u 03_train/main.py --name "$3" --set medium --embedder "$2" \
    --translation general --epochs 30 -y --verbose || return 1
}
run ps-sharedtiled yamnet_trunk_pitchshift_sharedtiled_depth12 v4-ft-ps-sharedtiled || F="$F sharedtiled"
run ps-untiled yamnet_trunk_pitchshift_untiled_depth12 v4-ft-ps-untiled || F="$F untiled"
say "QUEUE-DONE failed:${F:- none}"
[ -z "$F" ]
