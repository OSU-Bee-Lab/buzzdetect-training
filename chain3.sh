#!/usr/bin/env bash
# After chain2: train the shipped model for every run of this session, then deploy each to buzzdetect.
# Each step is independent (|| true); failures are listed at the end. Shipped epoch count is read from the
# fold curves (--skip-cv). Env must match the CV run's (LR, batch, fp16).
cd "$(dirname "$0")"
while kill -0 103340 2>/dev/null; do sleep 60; done
retry() { for i in 1 2 3 4 5 6; do "$@" && return 0; echo "[retry $i] $* exited nonzero; resuming (finished folds are skipped)"; sleep 5; done; return 1; }
PY="conda run --no-capture-output -n buzzdetect-train python -u"
export BUZZDETECT_CHUNK_FRAMES=48 TRUNK_FP16=1 TRUNK_LR_HEAD=2e-4
FAILED=""
ship() { # name embedder lr_backbone epochs batch [extra train args]
  n=$1; e=$2; lr=$3; ep=$4; b=$5; shift 5
  retry env TRUNK_BATCH=$b TRUNK_LR_BACKBONE=$lr $PY 03_train/main.py --name "$n" --set medium --embedder "$e" \
    --translation general --epochs "$ep" --skip-cv -y --verbose "$@" || { FAILED="$FAILED train:$n"; return; }
  $PY 04_deploy/main.py "$n" -y || FAILED="$FAILED deploy:$n"
}
ship trunk3-frozen        yamnet_trunk 0    60  1024
ship trunk3-ft-1e5        yamnet_trunk 1e-5 60  1024
ship trunk3-ft-3e5        yamnet_trunk 3e-5 60  1024
ship trunk3-ft-1e4        yamnet_trunk 1e-4 60  1024
ship trunk3-ft-1e5-e120   yamnet_trunk 1e-5 120 1024
ship trunk3-ft-3e5-do2    yamnet_trunk 3e-5 60  1024 --dropout 0.2
ship trunkctx-frozen      yamnet_trunk_context 0    120 512
ship trunkctx-ft-1e5      yamnet_trunk_context 1e-5 120 512
ship trunkctx-ft-3e5      yamnet_trunk_context 3e-5 120 512
echo "CHAIN3-DONE failed:${FAILED:- none}"
