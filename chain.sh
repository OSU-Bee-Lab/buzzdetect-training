#!/usr/bin/env bash
# ps-d8-ens-same5: two more plain ps-depth8 draws on medium (each resumes on rerun, finished folds skipped).
set -e
cd "$(dirname "$0")"
PY=$(bash tools/python_path.sh)
export TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024
for n in ps-depth8-r4 ps-depth8-r5; do
  $PY 03_train/main.py --name $n --set medium --embedder yamnet_trunk_pitchshift_depth8 --translation general --epochs 30 -y
done
