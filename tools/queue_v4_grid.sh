#!/usr/bin/env bash
# Era cv-medium-v4 opening queue (2026-09-25). Fire once through launch_job:
#
#   tools/launch_job.sh queue_v4_grid.log -- tools/queue_v4_grid.sh
#
# 1. cv-baseline-v4: the era anchor, frozen yamnet linear probe (default 400 epochs).
# 2. The retrain grid, every combination of {context, pitch-shift, dense head}, all
#    at the previous era's best settings: depth-12 cut (layers 12-14 trainable),
#    TRUNK_LR_BACKBONE=1e-5, 30 epochs, fp16. Dense = TRUNK_HIDDEN=1024 (ReLU).
#    (30, not last era's 60: trunkpsd12-ft-1e5-r2's cross-fold mean val sens is flat from
#    epoch 20 to 40, 0.378, and 0.374 at 60.)
# 3. moderate (the era's big-data set) extraction: yamnet and
#    yamnet_trunk_pitchshift_depth12. No training on it.
#
# Idempotent: rerunning the same command skips everything already done. Extraction
# is incremental, and a CV rerun skips finished folds. Each step retries 6 times
# (host-RAM restarts on the wide embedders are expected); a step that still fails
# is listed on the QUEUE-DONE line and the queue moves on to the next one.
# Grep the log for '^\[queue\]' to see where it is.
cd "$(dirname "$(realpath "$0")")/.."
source tools/python_path.sh

export BUZZDETECT_CHUNK_FRAMES=48 TRUNK_FP16=1 TRUNK_LR_HEAD=2e-4
FAILED=""
say() { echo "[queue] $(date '+%m-%d %H:%M') $*"; }
retry() {
  for i in 1 2 3 4 5 6; do
    "$@" && return 0
    say "retry $i: $* exited nonzero; resuming"
    sleep 5
  done
  return 1
}

extract() { # set embedder
  say "extract $1 $2"
  retry "$PY" -u 02_set/main.py --set "$1" --embedder "$2" --workers 1 --verbose \
    || FAILED="$FAILED extract:$1/$2"
}

train() { # name embedder epochs [env assignments...]
  local n=$1 e=$2 ep=$3; shift 3
  say "train $n ($e, $ep epochs, $*)"
  retry env "$@" "$PY" -u 03_train/main.py --name "$n" --set medium --embedder "$e" \
    --translation general --epochs "$ep" -y --verbose \
    || FAILED="$FAILED train:$n"
}

FT="TRUNK_LR_BACKBONE=1e-5"
DENSE="TRUNK_HIDDEN=1024"

# 1. anchor
extract medium yamnet
train cv-baseline-v4 yamnet 400

# 2. grid, cheapest first so the widest embedder (most likely to OOM) runs last
extract medium yamnet_trunk_depth12
train v4-ft              yamnet_trunk_depth12                    30 $FT TRUNK_BATCH=1024
train v4-ft-dense        yamnet_trunk_depth12                    30 $FT TRUNK_BATCH=1024 $DENSE

extract medium yamnet_trunk_pitchshift_depth12
train v4-ft-ps           yamnet_trunk_pitchshift_depth12         30 $FT TRUNK_BATCH=1024
train v4-ft-ps-dense     yamnet_trunk_pitchshift_depth12         30 $FT TRUNK_BATCH=1024 $DENSE

extract medium yamnet_trunk_context_depth12
train v4-ft-ctx          yamnet_trunk_context_depth12            30 $FT TRUNK_BATCH=512
train v4-ft-ctx-dense    yamnet_trunk_context_depth12            30 $FT TRUNK_BATCH=512 $DENSE

extract medium yamnet_trunk_pitchshift_context_depth12
train v4-ft-psctx        yamnet_trunk_pitchshift_context_depth12 30 $FT TRUNK_BATCH=512
train v4-ft-psctx-dense  yamnet_trunk_pitchshift_context_depth12 30 $FT TRUNK_BATCH=512 $DENSE

# 3. big-data set, extraction only
extract moderate yamnet
extract moderate yamnet_trunk_pitchshift_depth12

say "QUEUE-DONE failed:${FAILED:- none}"
[ -z "$FAILED" ]
