#!/usr/bin/env bash
# ps-d8-ktails: one CV of three ps-depth8 tails trained in one job on shared batches; the model's output
# is their logit mean, and each tail's own held-out predictions land in models/ps-d8-ktails3-t<k>.
# A rerun resumes (finished folds skipped), so a GPU OOM mid-CV is retried in a fresh process.
set -e
cd "$(dirname "$0")"
PY=$(bash tools/python_path.sh)
export TRUNK_KTAILS=3 TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 SCORE_CHUNK=256
NAME=${1:-ps-d8-ktails3}; SET=${2:-medium}; EPOCHS=${3:-30}
for attempt in 1 2 3 4; do
  if $PY 03_train/main.py --name $NAME --set $SET --embedder yamnet_trunk_pitchshift_depth8 --translation general --epochs $EPOCHS -y; then
    $PY 03_train/resummarize.py $NAME-t0 $NAME-t1 $NAME-t2
    exit 0
  fi
  echo "[chain] attempt $attempt failed; resuming"
done
exit 1
