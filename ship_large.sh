#!/usr/bin/env bash
# Train the shipped trunklarge-ft-1e5 model (epoch count read from the CV's fold
# curves) and deploy it to buzzdetect. Mirrors chain3.sh's ship() pattern.
cd "$(dirname "$0")"
retry() { for i in 1 2 3 4 5 6; do "$@" && return 0; echo "[retry $i] $* exited nonzero; resuming (finished folds are skipped)"; sleep 5; done; return 1; }
PY="conda run --no-capture-output -n buzzdetect-train python -u"
export BUZZDETECT_CHUNK_FRAMES=48 TRUNK_FP16=1 TRUNK_LR_HEAD=2e-4 TRUNK_BATCH=1024 TRUNK_LR_BACKBONE=1e-5
retry $PY 03_train/main.py --name trunklarge-ft-1e5 --set large --embedder yamnet_trunk \
  --translation general --epochs 60 --skip-cv -y --verbose || { echo "SHIPLARGE-FAILED: train"; exit 1; }
$PY 04_deploy/main.py trunklarge-ft-1e5 -y || { echo "SHIPLARGE-FAILED: deploy"; exit 1; }
echo SHIPLARGE-DONE
