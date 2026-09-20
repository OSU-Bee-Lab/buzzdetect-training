#!/usr/bin/env bash
# Runs after chain.sh: trunk fine-tune + temporal context, then longer-budget plain-trunk arms.
cd "$(dirname "$0")"
while kill -0 15431 2>/dev/null; do sleep 30; done
retry() { for i in 1 2 3 4 5 6; do "$@" && return 0; echo "[retry $i] $* exited nonzero; resuming (finished folds are skipped)"; sleep 5; done; return 1; }
PY="conda run --no-capture-output -n buzzdetect-train python -u"
export BUZZDETECT_CHUNK_FRAMES=48 TRUNK_FP16=1 TRUNK_LR_HEAD=2e-4
# context arms: 3 frames per sample, so batch 512 (~1536 frame-passes per step)
export TRUNK_BATCH=512
$PY 02_set/main.py --set lite --embedder yamnet_trunk_context --workers 1 || exit 1
retry env TRUNK_LR_BACKBONE=1e-5 $PY 03_train/main.py --name test_trunkctx_smoke --set lite --embedder yamnet_trunk_context --translation general --epochs 2 -y --verbose || exit 1
$PY 02_set/main.py --set medium --embedder yamnet_trunk_context --workers 1 || exit 1
run() { # name embedder lr_backbone epochs batch [extra]
  n=$1; e=$2; lr=$3; ep=$4; b=$5; shift 5
  retry env TRUNK_BATCH=$b TRUNK_LR_BACKBONE=$lr $PY 03_train/main.py --name "$n" --set medium --embedder "$e" \
    --translation general --epochs "$ep" -y --verbose "$@"
}
run trunkctx-frozen   yamnet_trunk_context 0    100 512
run trunkctx-ft-1e5   yamnet_trunk_context 1e-5 100 512
run trunkctx-ft-3e5   yamnet_trunk_context 3e-5 100 512
# longer budget on plain trunk, own matched control (a different budget is a different rule)
run trunk3-frozen-e200 yamnet_trunk 0    200 1024
run trunk3-ft-1e5-e200 yamnet_trunk 1e-5 200 1024
run trunk3-ft-3e5-e200 yamnet_trunk 3e-5 200 1024
echo CHAIN2-DONE
