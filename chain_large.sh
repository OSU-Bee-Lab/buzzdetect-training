#!/usr/bin/env bash
# trunk-ft-large: full 8-fold CV on `large` (framehop 0.2), YAMNet layers 13-14 fine-tuned.
# Settings mirror trunk3-ft-1e5 on medium (the control): 60 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024.
cd "$(dirname "$0")"
retry() { for i in 1 2 3 4 5 6; do "$@" && return 0; echo "[retry $i] $* exited nonzero; resuming (finished folds are skipped)"; sleep 5; done; return 1; }
PY="conda run --no-capture-output -n buzzdetect-train python -u"
export BUZZDETECT_CHUNK_FRAMES=48 TRUNK_FP16=1 TRUNK_LR_HEAD=2e-4 TRUNK_BATCH=1024 TRUNK_LR_BACKBONE=1e-5
retry $PY 03_train/main.py --name trunklarge-ft-1e5 --set large --embedder yamnet_trunk \
  --translation general --epochs 60 -y --verbose || { echo "CHAINLARGE-FAILED: retries exhausted"; exit 1; }
echo CHAINLARGE-DONE
