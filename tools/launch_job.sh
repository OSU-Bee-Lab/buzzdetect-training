#!/usr/bin/env bash
# Launch a long job detached -- the only launch that survives Claude Code (see
# CLAUDE.md "Running long jobs"). It prints the job's pid and the Monitor
# command to watch it with (tools/watch_job.sh).
#
#   tools/launch_job.sh [--cpu] <log> -- <command...>
#
#   tools/launch_job.sh train_x.log -- 03_train/main.py --name x --set medium -y
#   tools/launch_job.sh extract.log -- 02_set/main.py --set medium --embedder e --workers 1
#   tools/launch_job.sh dl.log -- curl -fLo embedders/e/weights.bin https://...
#
# A .py command runs under the buzzdetect-train python, unbuffered, with
# MALLOC_ARENA_MAX=2. Stage 2 also gets BUZZDETECT_CHUNK_FRAMES=48, and any
# pipeline main.py gets --verbose. Other commands run as given. Env vars set by
# the caller pass through.
#
# The GPU is visible by default. --cpu hides it (CUDA_VISIBLE_DEVICES=); use it
# to re-run a job that ran out of GPU memory (stage 3 skips the folds already
# finished and stage 2 resumes per ident). The 4 GB card fits one GPU job at a
# time; don't launch a second alongside it.
# --gpu is accepted and does nothing (it was the opt-in before GPU became the
# default).
#
# The log is overwritten. Every line the job prints is prefixed with the wall
# clock ("09-14 13:52:07 "); the log ends with an unprefixed
# "[launch_job] exit N", the job's exit status.

# Run the main checkout's copy: a worktree's tools/ is frozen at its branch point.
_main="$(dirname "$(git -C "$(dirname "$(realpath "$0")")" rev-parse --path-format=absolute --git-common-dir)")/tools/$(basename "$0")"
[ "$(realpath "$0")" = "$(realpath -m "$_main")" ] || [ ! -f "$_main" ] || exec bash "$_main" "$@"

set -euo pipefail

PY=/home/luke/anaconda3/envs/buzzdetect-train/bin/python
usage="usage: launch_job.sh [--cpu] <log> -- <command...>"
cpu=0
while [ $# -gt 0 ]; do
  case $1 in
    --cpu) cpu=1; shift ;;
    --gpu) shift ;;
    *) break ;;
  esac
done
log=${1:?$usage}; shift
[ "${1:-}" = -- ] && shift
[ $# -gt 0 ] || { echo "$usage" >&2; exit 2; }

envs=(PYTHONUNBUFFERED=1)
[ "$cpu" = 1 ] && envs+=(CUDA_VISIBLE_DEVICES=)
cmd=("$@")
if [[ $1 == *.py ]]; then
  envs+=(MALLOC_ARENA_MAX=2)
  [[ $1 == *02_set/main.py ]] && envs+=("BUZZDETECT_CHUNK_FRAMES=${BUZZDETECT_CHUNK_FRAMES:-48}")
  [[ $1 == *main.py && " $* " != *" --verbose "* ]] && cmd+=(--verbose)
  cmd=("$PY" -u "${cmd[@]}")
fi

log=$(realpath -m "$log")
# setsid makes the job its own process group (pgid = pid), so one signal to the
# group reaches every worker it spawned
nohup setsid env "${envs[@]}" bash -c '
  "$@" 2>&1 | while IFS= read -r line; do printf "%(%m-%d %T)T %s\n" -1 "$line"; done
  echo "[launch_job] exit ${PIPESTATUS[0]}"' launch_job "${cmd[@]}" \
  > "$log" 2>&1 < /dev/null &
pid=$!
disown

# Register it in the main checkout (shared by every worktree), so
# tools/human/agent_loop.sh can kill the jobs its agents started.
common=$(git -C "$(dirname "$(realpath "$0")")" rev-parse --path-format=absolute --git-common-dir)
jobs_dir="$(dirname "$common")/.local/jobs"
mkdir -p "$jobs_dir"
echo "$log :: ${cmd[*]}" > "$jobs_dir/$pid"

sleep 5
if ! kill -0 "$pid" 2>/dev/null; then
  echo "job exited within 5 s; log tail:" >&2
  tail -n 20 "$log" >&2
  exit 1
fi
echo "pid $pid · log $log"
echo "watch it: Monitor, timeout_ms 1800000, re-armed at every expiry, command: $(dirname "$(realpath "$0")")/watch_job.sh $pid $log"
