#!/usr/bin/env bash
# The Monitor command for any detached job: a CV, an extraction, a weights
# download, a one-off script. Arm it via Monitor with timeout_ms 1800000. Every
# event is a header line followed by the job log's last 3 lines (timestamped by
# launch_job.sh):
#
#   progress 3/8 · watch continues          one per unit of work (a stage-3 fold)
#   ERROR in log (job still running) · watch continues     once per new error line
#   still running · 1h32m · watch ended: re-arm this command    after WINDOW
#   DONE / FAILED (exit N) / EXITED                          the job ended
#
# The watcher ends its own watch after WINDOW (1740 s), just inside Monitor's
# 30-min cap, so a watch never dies to a bare expiry: the "re-arm" line is the
# one event to act on while the job runs, and it doubles as the cache keepalive
# (the prompt cache's TTL refreshes on every read, so wakes under an hour apart
# hold it warm through a run of any length).
#
# Stage 2 has no progress unit on purpose: one ping per ident is too many. An
# extraction reports only errors, the re-arm line and its end. There is no ETA
# either: per-unit cost varies too much for an extrapolation to mean anything.
# If a job looks wrong, read its log.
#
#   tools/watch_job.sh <pid> [--log FILE] [--progress REGEX] [--total N]
#                            [--size PATH] [--until CMD]
#
#   --log       the job's log: progress, errors, exit status, the lines shown
#   --progress  a log line matching this is one unit of work. Default: stage 3's
#               "[k/N] <fold>:" line, printed once per fold
#   --total     units in the job, for the k/N count, when the lines don't carry
#               [k/N]
#   --size      report this path's size on the re-arm line (a download)
#   --until     a shell condition; exit DONE when it succeeds. Pass pid "-" to
#               watch a condition with no process behind it
#
# The exit status comes from the "[launch_job] exit N" line tools/launch_job.sh
# appends; a job launched another way reports EXITED, status unknown.
set -uo pipefail

pid=${1:?usage: watch_job.sh <pid|-> [--log FILE] [--progress REGEX] [--total N] [--size PATH] [--until CMD]}
shift
log=""; progress='\[[0-9]+/[0-9]+\] [^ ].*: '; total=""; size=""; until_cmd=""
while [ $# -gt 0 ]; do
  case $1 in
    --log) log=$2 ;;
    --progress) progress=$2 ;;
    --total) total=$2 ;;
    --size) size=$2 ;;
    --until) until_cmd=$2 ;;
    *) echo "watch_job.sh: unknown argument $1" >&2; exit 2 ;;
  esac
  shift 2
done
WINDOW=${WINDOW:-1740}
POLL=${POLL:-30}
ERRORS='Traceback|MemoryError|Killed|Segmentation fault|OutOfMemory|out of memory'

count() { local n=""; [ -n "$log" ] && n=$(grep -cE "$1" "$log" 2>/dev/null); echo "${n:-0}"; }

# a header line, then the log's last 3 lines, in one write so Monitor delivers
# them as one notification
event() {
  local tail_lines=""
  [ -n "$log" ] && tail_lines=$(grep -v '^\[launch_job\] exit' "$log" 2>/dev/null | tail -n 3 | cut -c1-200 | sed 's/^/  /')
  printf '%s\n%s\n' "$1" "$tail_lines"
}

start=$(date +%s)
units=$(count "$progress")
errors=$(count "$ERRORS")
while true; do
  c=$(count "$progress")
  if [ "$c" -gt "$units" ]; then
    units=$c
    k=$units; n=$total
    line=$(grep -E "$progress" "$log" | tail -n 1)
    [[ $line =~ \[([0-9]+)/([0-9]+)\] ]] && { k=${BASH_REMATCH[1]}; n=${BASH_REMATCH[2]}; }
    event "progress $k${n:+/$n} · watch continues"
  fi

  c=$(count "$ERRORS")
  if [ "$c" -gt "$errors" ]; then
    errors=$c
    event "ERROR in log (job still running): $(grep -E "$ERRORS" "$log" | tail -n 1 | cut -c1-160) · watch continues"
  fi

  if [ -n "$until_cmd" ] && bash -c "$until_cmd" >/dev/null 2>&1; then
    event "DONE: condition met: $until_cmd"; exit 0
  fi

  if [ "$pid" != - ] && ! kill -0 "$pid" 2>/dev/null; then
    sleep 2  # let the last lines land
    status=$(grep -oE '^\[launch_job\] exit [0-9]+' "$log" 2>/dev/null | tail -n 1 | grep -oE '[0-9]+$')
    if [ "$status" = 0 ]; then
      event "DONE: exit 0"
    elif [ -n "$status" ]; then
      event "FAILED (exit $status) · $(grep -E "$ERRORS" "$log" | tail -n 1 | cut -c1-160) · see $log"
    else
      event "EXITED, status unknown"
    fi
    exit 0
  fi

  now=$(date +%s)
  if [ $(( now - start )) -ge "$WINDOW" ]; then
    e=""; [ "$pid" != - ] && e=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
    e=${e:-$(( now - start ))}
    event "still running · $((e / 3600))h$(( e % 3600 / 60 ))m${size:+ · $(du -sh "$size" 2>/dev/null | cut -f1) at $size} · watch ended: re-arm this command"
    exit 0
  fi
  sleep "$POLL"
done
