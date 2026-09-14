#!/usr/bin/env bash
# Launch a long job detached -- the only launch that survives Claude Code (see
# CLAUDE.md "Running long jobs") -- then print its PID and the watch_job.sh
# command to arm as a persistent Monitor.
#
#   tools/launch_job.sh [--gpu] <log> -- <command...>
#
#   tools/launch_job.sh train_x.log -- 03_train/main.py --name x --set medium -y
#   tools/launch_job.sh extract.log -- 02_set/main.py --set medium --embedder e --workers 1
#   tools/launch_job.sh dl.log -- curl -fLo embedders/e/weights.bin https://...
#
# A .py command runs under the buzzdetect-train python, unbuffered, with
# MALLOC_ARENA_MAX=2 and the GPU hidden (the 12288-d trunk embedders OOM the
# 4 GB card, and CPU ~ GPU for the probe). Stage 2 also gets
# BUZZDETECT_CHUNK_FRAMES=48, and any pipeline main.py gets --verbose. --gpu
# leaves CUDA visible, for an embedder that manages the GPU itself (AVES wants
# BUZZDETECT_NO_GPU=1 set by the caller instead). Other commands run as given.
# Env vars set by the caller pass through.
#
# The log is overwritten and ends with "[launch_job] exit N", which is how
# watch_job.sh tells DONE from FAILED.
set -euo pipefail

PY=/home/luke/anaconda3/envs/buzzdetect-train/bin/python
usage="usage: launch_job.sh [--gpu] <log> -- <command...>"
gpu=0
[ "${1:-}" = --gpu ] && { gpu=1; shift; }
log=${1:?$usage}; shift
[ "${1:-}" = -- ] && shift
[ $# -gt 0 ] || { echo "$usage" >&2; exit 2; }

envs=(PYTHONUNBUFFERED=1)
cmd=("$@")
if [[ $1 == *.py ]]; then
  envs+=(MALLOC_ARENA_MAX=2)
  [ "$gpu" = 1 ] || envs+=(CUDA_VISIBLE_DEVICES=)
  [[ $1 == *02_set/main.py ]] && envs+=("BUZZDETECT_CHUNK_FRAMES=${BUZZDETECT_CHUNK_FRAMES:-48}")
  [[ $1 == *main.py && " $* " != *" --verbose "* ]] && cmd+=(--verbose)
  cmd=("$PY" -u "${cmd[@]}")
fi

log=$(realpath -m "$log")
nohup env "${envs[@]}" bash -c '"$@"; echo "[launch_job] exit $?"' launch_job "${cmd[@]}" \
  > "$log" 2>&1 < /dev/null &
pid=$!
disown

sleep 5
if ! kill -0 "$pid" 2>/dev/null; then
  echo "job exited within 5 s; log tail:" >&2
  tail -n 20 "$log" >&2
  exit 1
fi
echo "pid $pid · log $log"
echo "Monitor (persistent: true): $(dirname "$(realpath "$0")")/watch_job.sh $pid --log '$log'"
