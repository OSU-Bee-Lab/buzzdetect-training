#!/usr/bin/env bash
# ens3-moderate: two more draws of the teacher's recipe on moderate (tools/queue_v4_moderate.sh's train(),
# 5 rotations each, no shipped model; each resumes on rerun, finished folds skipped), then the logit means.
set -e
cd "$(dirname "$0")"
PY=$(bash tools/python_path.sh)
export BUZZDETECT_CHUNK_FRAMES=48 TRUNK_FP16=1 TRUNK_LR_HEAD=2e-4 TRUNK_STREAM=1 TRUNK_LR_BACKBONE=1e-5 TRUNK_BATCH=1024
for n in v4-ft-ps-e60-moderate-r2 v4-ft-ps-e60-moderate-r3; do
  $PY -u 03_train/main.py --name $n --set moderate --embedder yamnet_trunk_pitchshift_depth12 \
    --translation general --epochs 60 -y --verbose
done
A=/home/luke/projects/buzzdetect-training/models/v4-ft-ps-e60-moderate
B=models/v4-ft-ps-e60-moderate-r2
C=models/v4-ft-ps-e60-moderate-r3
$PY tools_exp/ensemble_predictions.py ens3-moderate $A $B $C
$PY tools_exp/ensemble_predictions.py ens2ab-moderate $A $B
$PY tools_exp/ensemble_predictions.py ens2ac-moderate $A $C
$PY tools_exp/ensemble_predictions.py ens2bc-moderate $B $C
