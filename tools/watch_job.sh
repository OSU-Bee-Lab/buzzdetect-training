#!/usr/bin/env bash
# The command for Claude Code's Monitor tool on a tools/launch_job.sh job. Arm it
# with timeout_ms 1800000 (the maximum) and re-arm it at every expiry until the
# job ends (CLAUDE.md "Running long jobs").
#
#   tools/watch_job.sh <pid> <log>
#
# Its first line says where the job stands, so each re-arm doubles as a check
# and covers anything logged while no Monitor was armed:
#   <now> running · <k/N folds> · <n> error lines · last: <last log line>
#   <now> not running · <same counts> · [launch_job] exit N
# Then it stays silent until something completes: stage 3's CV headline
# ("[<name>] sens@fpr... per deployment: ..."), the shipped model
# ("[shipped] ..."), and the log's closing "[launch_job] exit N", each but the
# last with launch_job's timestamp. It exits when the job does, so the exit line
# is always its last event. A crash ends the job, so it arrives as that exit
# line. An error that leaves the job hanging shows only in the error count on
# the next re-arm.

# Run the main checkout's copy: a worktree's tools/ is frozen at its branch point.
_main="$(dirname "$(git -C "$(dirname "$(realpath "$0")")" rev-parse --path-format=absolute --git-common-dir)")/tools/$(basename "$0")"
[ "$(realpath "$0")" = "$(realpath -m "$_main")" ] || [ ! -f "$_main" ] || exec bash "$_main" "$@"

usage="usage: watch_job.sh <pid> <log>"
pid=${1:?$usage}; log=${2:?$usage}

# Anchored to launch_job's timestamp prefix: Keras progress bars and TF's
# allocator warnings otherwise match error words mid-line.
EVENTS='^\[launch_job\] exit|^[0-9-]+ [0-9:]+ (\[[^]]+\] sens@fpr[^ ]* per deployment: |\[shipped\] )'
ERRORS='^[0-9-]+ [0-9:]+ (Traceback|[A-Za-z_.]*(Error|Exception)( |:|$)|.*line [0-9]+: +[0-9]+ (Killed|Segmentation fault))'
fold=$(grep -oE '^[0-9-]+ [0-9:]+ \[[0-9]+/[0-9]+\] [^ ].*: ' "$log" 2>/dev/null | tail -n 1 | grep -oE '[0-9]+/[0-9]+')
counts="${fold:-0} folds · $(grep -cE "$ERRORS" "$log" 2>/dev/null || true) error lines"
now=$(date '+%m-%d %T')

if ! kill -0 "$pid" 2>/dev/null; then
  status=$(grep -E '^\[launch_job\] exit' "$log" 2>/dev/null | tail -n 1)
  echo "$now not running · $counts · ${status:-no [launch_job] exit line}"
  exit 0
fi
echo "$now running · $counts · last: $(tail -n 1 "$log" | tr -d '\r\033' | cut -c1-160)"
tail --pid="$pid" -n 0 -F "$log" 2>/dev/null | grep --line-buffered -E "$EVENTS"
