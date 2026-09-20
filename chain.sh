#!/usr/bin/env bash
# Weekend queue: YAMNet trunk fine-tuning (layers 13-14), medium set. Resumable: re-run to continue.
cd "$(dirname "$0")"
PY="conda run --no-capture-output -n buzzdetect-train python -u"
export BUZZDETECT_CHUNK_FRAMES=48 TRUNK_FP16=1 TRUNK_BATCH=1024 TRUNK_LR_HEAD=2e-4
# smoke test on lite; abort the whole chain if it breaks
$PY 02_set/main.py --set lite --embedder yamnet_trunk --workers 1 || exit 1
TRUNK_LR_BACKBONE=1e-5 $PY 03_train/main.py --name test_trunk_smoke --set lite --embedder yamnet_trunk --translation general --epochs 2 -y --verbose || exit 1
$PY 02_set/main.py --set medium --embedder yamnet_trunk --workers 1 || exit 1
run() { # name lr_backbone epochs [extra args]
  n=$1; lr=$2; ep=$3; shift 3
  TRUNK_LR_BACKBONE=$lr $PY 03_train/main.py --name "$n" --set medium --embedder yamnet_trunk \
    --translation general --epochs "$ep" -y --verbose "$@"
}
run trunk3-frozen   0    60
run trunk3-ft-1e5   1e-5 60
run trunk3-ft-3e5   3e-5 60
run trunk3-ft-1e4   1e-4 60
run trunk3-ft-1e5-e120 1e-5 120
run trunk3-ft-3e5-do2  3e-5 60 --dropout 0.2
echo CHAIN-DONE
