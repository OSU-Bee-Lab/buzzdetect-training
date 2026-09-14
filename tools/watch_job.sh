#!/usr/bin/env bash
# The Monitor command for any detached job: a CV, an extraction, a weights
# download, a one-off script. Run it via Monitor with `persistent: true` (a timed
# Monitor caps at 1 h). Every line it prints wakes the agent:
#
#   progress 3/8 · 6.2 min each · ~31 min left (ETA 14:05) · <log line>
#   ERROR in log (job still running): <line>        once per new error line
#   still running · 1h32m · <last log line>          no other event for 45 min
#   DONE / FAILED (exit N) / EXITED                  terminal; the watcher exits
#
#   tools/watch_job.sh <pid> [--log FILE] [--progress REGEX] [--total N]
#                            [--size PATH] [--until CMD]
#
#   --log       the job's log: progress, errors, exit status, heartbeat context
#   --progress  a log line matching this is one unit of work. Default: stage 3's
#               "[k/N] <fold>:" and stage 2's "(ident took" lines
#   --total     units in the job, for the ETA, when the lines don't carry [k/N].
#               Default: stage 2's "<N> to extract" line, if any
#   --size      report this path's size in each heartbeat (a download)
#   --until     a shell condition; exit DONE when it succeeds. Pass pid "-" to
#               watch a condition with no process behind it
#
# The heartbeat is a cache keepalive: the prompt cache's TTL refreshes on every
# read, so wakes under an hour apart hold it warm through a run of any length.
# The exit status comes from the "[launch_job] exit N" line tools/launch_job.sh
# appends; a job launched another way reports EXITED, status unknown.
set -uo pipefail

pid=${1:?usage: watch_job.sh <pid|-> [--log FILE] [--progress REGEX] [--total N] [--size PATH] [--until CMD]}
shift
log=""; progress='^\[[0-9]+/[0-9]+\] |\(ident took'; total=""; size=""; until_cmd=""
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
HEARTBEAT=${HEARTBEAT:-2700}
POLL=${POLL:-30}
ERRORS='Traceback|MemoryError|Killed|Segmentation fault|OutOfMemory|out of memory'

count() { local n=""; [ -n "$log" ] && n=$(grep -cE "$1" "$log" 2>/dev/null); echo "${n:-0}"; }
last_line() { tail -n 1 "$log" 2>/dev/null | cut -c1-160; }
minutes() { awk -v s="$1" 'BEGIN { printf "%.1f", s / 60 }'; }

progress_event() {
  local line k="" n="" now per left
  line=$(grep -E "$progress" "$log" | tail -n 1)
  if [[ $line =~ \[([0-9]+)/([0-9]+)\] ]]; then
    k=${BASH_REMATCH[1]}; n=${BASH_REMATCH[2]}
  else
    k=$units
    n=${total:-$(grep -oE '[0-9]+ to extract' "$log" 2>/dev/null | tail -n 1 | cut -d' ' -f1)}
  fi
  now=$(date +%s)
  per=$(( (now - start) / (units - units0) ))
  if [ -n "$n" ]; then
    left=$(( (n - k) * per ))
    echo "progress $k/$n · $(minutes $per) min each · ~$(minutes $left) min left (ETA $(date -d "@$((now + left))" +%H:%M)) · ${line:0:120}"
  else
    echo "progress $k · $(minutes $per) min each · ${line:0:120}"
  fi
}

start=$(date +%s); last_event=$start
units0=$(count "$progress"); units=$units0
errors=$(count "$ERRORS")
while true; do
  c=$(count "$progress")
  if [ "$c" -gt "$units" ]; then units=$c; last_event=$(date +%s); progress_event; fi

  c=$(count "$ERRORS")
  if [ "$c" -gt "$errors" ]; then
    errors=$c; last_event=$(date +%s)
    echo "ERROR in log (job still running): $(grep -E "$ERRORS" "$log" | tail -n 1 | cut -c1-160)"
  fi

  if [ -n "$until_cmd" ] && bash -c "$until_cmd" >/dev/null 2>&1; then
    echo "DONE: condition met: $until_cmd"; exit 0
  fi

  if [ "$pid" != - ] && ! kill -0 "$pid" 2>/dev/null; then
    sleep 2  # let the last lines land
    status=$(grep -oE '^\[launch_job\] exit [0-9]+' "$log" 2>/dev/null | tail -n 1 | grep -oE '[0-9]+$')
    if [ "$status" = 0 ]; then
      echo "DONE: exit 0 · $(tail -n 2 "$log" | head -n 1 | cut -c1-160)"
    elif [ -n "$status" ]; then
      echo "FAILED (exit $status) · $(grep -E "$ERRORS" "$log" | tail -n 1 | cut -c1-160) · see $log"
    else
      echo "EXITED, status unknown · last: $(last_line)"
    fi
    exit 0
  fi

  if [ $(( $(date +%s) - last_event )) -ge "$HEARTBEAT" ]; then
    last_event=$(date +%s)
    e=$(( last_event - start ))
    echo "still running · $((e / 3600))h$(( e % 3600 / 60 ))m${size:+ · $(du -sh "$size" 2>/dev/null | cut -f1) at $size}${log:+ · $(last_line)}"
  fi
  sleep "$POLL"
done
