#!/usr/bin/env bash
# The command for Claude Code's Monitor tool on tools/launch_job.sh jobs. Arm it
# with timeout_ms 1800000 (the maximum) and re-arm it at every expiry until the
# jobs end (CLAUDE.md "Running long jobs").
#
#   tools/watch_job.sh                     every job this session launched
#   tools/watch_job.sh <pid> <log> [...]   these jobs only (another session's)
#   tools/watch_job.sh --adopt <pid>       make another session's job yours
#
# With no arguments it watches every job launch_job.sh started from this
# session (same $CLAUDE_JOB_DIR), including ones launched after it was armed, so
# one Monitor covers them all: one expiry per 30 min however many jobs run, and
# no re-arm when a new job starts. --adopt moves a job some other session
# started (a HANDOFF.md resume) into this session's set; a Monitor already
# armed picks it up.
#
# Its first lines say where each job stands, so each re-arm doubles as a check
# and covers anything logged while no Monitor was armed:
#   <now> running · <k/N folds> · <n> error lines · last: <last log line>
#   <now> not running · <same counts> · [launch_job] exit N
# (a job that ended unseen is reported once; a job launched later gets a
# "started" line). Then it stays silent until something completes: stage 3's
# CV headline ("[<name>] sens@fpr... per deployment: ..."), the shipped model
# ("[shipped] ..."), and the log's closing "[launch_job] exit N". With more than
# one job, each line is prefixed by its log's name. It exits when the last job
# it follows does. A crash ends a job, so it arrives as that job's exit line. An
# error that leaves a job hanging shows only in the error count on the next
# re-arm.

# Run the main checkout's copy: a worktree's tools/ is frozen at its branch point.
_main="$(dirname "$(git -C "$(dirname "$(realpath "$0")")" rev-parse --path-format=absolute --git-common-dir)")/tools/$(basename "$0")"
[ ! -f "$_main" ] || [ "$(realpath "$0")" = "$(realpath "$_main")" ] || exec bash "$_main" "$@"

usage="usage: watch_job.sh [<pid> <log>]... | --adopt <pid>"
common=$(git -C "$(dirname "$(realpath "$0")")" rev-parse --path-format=absolute --git-common-dir)
JOBS="$(dirname "$common")/.local/jobs"   # launch_job.sh's registry

# Anchored to launch_job's timestamp prefix: Keras progress bars and TF's
# allocator warnings otherwise match error words mid-line.
EVENTS='^\[launch_job\] exit|^[0-9-]+ [0-9:]+ (\[[^]]+\] sens@fpr[^ ]* per deployment: |\[shipped\] |\[queue\] )'
ERRORS='^[0-9-]+ [0-9:]+ (Traceback|[A-Za-z_.]*(Error|Exception)( |:|$)|.*line [0-9]+: +[0-9]+ (Killed|Segmentation fault))'

state() {  # <pid> <log> -> one line: where the job stands
  local fold counts now status
  fold=$(grep -oE '^[0-9-]+ [0-9:]+ \[[0-9]+/[0-9]+\] [^ ].*: ' "$2" 2>/dev/null | tail -n 1 | grep -oE '[0-9]+/[0-9]+')
  counts="${fold:-0} folds · $(grep -cE "$ERRORS" "$2" 2>/dev/null || true) error lines"
  now=$(date '+%m-%d %T')
  if ! kill -0 "$1" 2>/dev/null; then
    status=$(grep -E '^\[launch_job\] exit' "$2" 2>/dev/null | tail -n 1)
    echo "$now not running · $counts · ${status:-no [launch_job] exit line}"
  else
    echo "$now running · $counts · last: $(tail -n 1 "$2" | tr -d '\r\033' | cut -c1-160)"
  fi
}
follow() {  # <pid> <log> -> its events until it exits
  if tail --version 2>/dev/null | grep -q GNU; then
    tail --pid="$1" -n 0 -F "$2" 2>/dev/null | grep --line-buffered -E "$EVENTS"
    return
  fi
  # BSD tail (macOS) has no --pid: follow until the job is gone, give tail a
  # beat to read the exit line, then stop it so grep sees EOF.
  local tp
  tail -n 0 -F "$2" 2>/dev/null > >(grep --line-buffered -E "$EVENTS") &
  tp=$!
  while kill -0 "$1" 2>/dev/null; do sleep 2; done
  sleep 2
  kill "$tp" 2>/dev/null
  wait "$tp" 2>/dev/null
  sleep 0.2  # let grep flush its last line
}

# One job: unprefixed, as HANDOFF.md files written before 2026-09-23 expect.
if [ $# = 2 ] && [ "$1" != --adopt ]; then
  line=$(state "$1" "$2"); echo "$line"
  [[ $line == *" not running "* ]] || follow "$1" "$2"
  exit 0
fi

need_session() {
  [ -n "${CLAUDE_JOB_DIR:-}" ] || { echo "watch_job: no \$CLAUDE_JOB_DIR, so no session to watch; pass <pid> <log> pairs" >&2; exit 2; }
}
if [ "${1:-}" = --adopt ]; then
  need_session
  f="$JOBS/${2:?$usage}"
  [ -f "$f" ] || { echo "watch_job: no launch_job.sh job $2 in $JOBS" >&2; exit 1; }
  # Keep the mtime: tools/human/agent_loop.sh kills the jobs registered since
  # a session started, and an adopted job was not started by this one.
  ref=$(mktemp); touch -r "$f" "$ref"   # portable mtime save (no GNU stat -c)
  printf '%s\nsession %s\n' "$(head -n 1 "$f")" "$CLAUDE_JOB_DIR" > "$f"
  touch -r "$ref" "$f"; rm -f "$ref"
  echo "adopted job $2 ($(sed -n '1s/ :: .*//p' "$f")); tools/watch_job.sh with no arguments now follows it"
  exit 0
fi

if [ $# -gt 0 ]; then
  [ $(( $# % 2 )) = 0 ] || { echo "$usage" >&2; exit 2; }
  pairs=("$@")
  listed() { printf '%s %s\n' "${pairs[@]}"; }
else
  need_session
  # A job that ended between Monitors is reported once, then left out.
  reported="$CLAUDE_JOB_DIR/watch_job.reported"; touch "$reported"
  listed() {
    local f
    for f in "$JOBS"/*; do
      [ -f "$f" ] && grep -qxF "session $CLAUDE_JOB_DIR" "$f" || continue
      grep -qxF "${f##*/}" "$reported" && continue
      echo "${f##*/} $(sed -n '1s/ :: .*//p' "$f")"
    done
  }
fi

declare -A followed
first=1
while :; do
  while read -r pid log; do
    [ -n "${followed[$pid]:-}" ] && continue
    followed[$pid]=1
    name=$(basename "$log" .log)
    line=$(state "$pid" "$log")
    if [[ $line == *" not running "* ]]; then
      echo "$name: $line"
      [ -n "${reported:-}" ] && echo "$pid" >> "$reported"
    else
      echo "$name: $([ $first = 1 ] || echo 'started · ')$line"
      { follow "$pid" "$log" | sed -u "s|^|$name: |"
        [ -n "${reported:-}" ] && echo "$pid" >> "$reported"; } &
    fi
  done < <(listed)
  if [ $first = 1 ] && [ ${#followed[@]} = 0 ]; then
    echo "$(date '+%m-%d %T') no running jobs launched from this session"
    exit 0
  fi
  first=0
  [ -n "$(jobs -rp)" ] || exit 0
  [ $# -gt 0 ] && { wait; exit 0; }   # a fixed set: nothing new can join
  sleep 5
done
