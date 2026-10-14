#!/usr/bin/env bash
# d1-ens-teacher, step 1: shipped models for draws r2 and r3 of the teacher's recipe (their rotations came
# from exp/ens3-moderate), as tools/queue_v4_moderate_ship.sh shipped draw 1, then each one's ONNX into a
# scratch directory (not buzzdetect: only the ensemble graph is deployed there). Resumes on rerun.
set -e
cd "$(dirname "$(realpath "$0")")"
PY=$(bash tools/python_path.sh)
export BUZZDETECT_CHUNK_FRAMES=48 TRUNK_FP16=1 TRUNK_LR_HEAD=2e-4 TRUNK_STREAM=1 TRUNK_LR_BACKBONE=1e-5 TRUNK_BATCH=1024
DEST=$PWD/.local/members
mkdir -p "$DEST"
retry() {
  for i in 1 2 3 4; do
    "$@" && return 0
    echo "[ship] retry $i: $* exited nonzero; resuming"
    sleep 5
  done
  return 1
}
for n in v4-ft-ps-e60-moderate-r2 v4-ft-ps-e60-moderate-r3; do
  if [ ! -f "models/$n/model.keras" ]; then
    retry "$PY" -u 03_train/main.py --name "$n" --set moderate --embedder yamnet_trunk_pitchshift_depth12 \
      --translation general --epochs 60 --skip-cv -y --verbose
  fi
  if [ ! -f "$DEST/$n/model.onnx" ]; then
    "$PY" -u 04_deploy/main.py "$n" --dest "$DEST" -y
  fi
done
echo "[ship] SHIP-DONE"
