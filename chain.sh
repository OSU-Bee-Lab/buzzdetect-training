#!/usr/bin/env bash
set -e
cd "$(dirname "$(realpath "$0")")"
PY=$(bash tools/python_path.sh)
BUZZDETECT_CHUNK_FRAMES=48 $PY 02_set/main.py --set medium --embedder yamnet_trunk_pitchshift2_depth8 --workers 1
TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 $PY 03_train/main.py --name ps2-depth8 --set medium \
  --embedder yamnet_trunk_pitchshift2_depth8 --translation general --epochs 30 -y
