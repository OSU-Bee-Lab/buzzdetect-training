#!/usr/bin/env bash
# Linear-probe anchor on `moderate` (Luke, 2026-09-28): separates how much of the
# v4-ft-*-moderate gain comes from the bigger training pool vs from the trunk
# fine-tune. Matched to cv-baseline-v4 (medium: 0.324): frozen yamnet, bare
# linear probe, 400 epochs, --translation general.
#
#   tools/launch_job.sh queue_v4_moderate_baseline.log -- tools/queue_v4_moderate_baseline.sh
#
# The extract step is a no-op when moderate's yamnet embeddings are complete; it
# is here in case they predate Hard Negatives. Idempotent and retrying, same as
# queue_v4_moderate.sh. Grep '^\[queue\]'.
cd "$(dirname "$(realpath "$0")")/.."
source tools/python_path.sh

export BUZZDETECT_CHUNK_FRAMES=48
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

say "extract moderate yamnet"
retry "$PY" -u 02_set/main.py --set moderate --embedder yamnet --workers 1 --verbose \
  || FAILED="$FAILED extract:moderate/yamnet"

say "train cv-baseline-v4-moderate (yamnet, 400 epochs)"
retry "$PY" -u 03_train/main.py --name cv-baseline-v4-moderate --set moderate --embedder yamnet \
  --translation general --epochs 400 -y --verbose \
  || FAILED="$FAILED train:cv-baseline-v4-moderate"

say "QUEUE-DONE failed:${FAILED:- none}"
[ -z "$FAILED" ]
