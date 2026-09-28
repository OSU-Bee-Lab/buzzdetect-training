#!/usr/bin/env bash
# ps-fast: lite smoke (extract + 2-epoch CV), then medium extract + 30-epoch CV
# matched to v4-ft-ps. Idempotent and retrying; grep '^\[queue\]'.
cd "$(dirname "$(realpath "$0")")"
source /home/luke/projects/buzzdetect-training/tools/python_path.sh

export BUZZDETECT_CHUNK_FRAMES=48 TRUNK_FP16=1 TRUNK_LR_HEAD=2e-4
E=yamnet_trunk_pitchshift_fast_depth12
say() { echo "[queue] $(date '+%m-%d %H:%M') $*"; }
retry() {
  for i in 1 2 3; do
    "$@" && return 0
    say "retry $i: $* exited nonzero; resuming"
    sleep 5
  done
  say "FAILED: $*"; exit 1
}

say "smoke: extract lite"
retry "$PY" -u 02_set/main.py --set lite --embedder $E --workers 1 --verbose
say "smoke: train lite, 2 epochs"
retry env TRUNK_LR_BACKBONE=1e-5 TRUNK_BATCH=1024 "$PY" -u 03_train/main.py --name test_ps_fast_smoke \
  --set lite --embedder $E --translation general --epochs 2 -y --verbose
say "extract medium"
retry "$PY" -u 02_set/main.py --set medium --embedder $E --workers 1 --verbose
say "train v4-ft-ps-fast"
retry env TRUNK_LR_BACKBONE=1e-5 TRUNK_BATCH=1024 "$PY" -u 03_train/main.py --name v4-ft-ps-fast \
  --set medium --embedder $E --translation general --epochs 30 -y --verbose
say "QUEUE-DONE"
