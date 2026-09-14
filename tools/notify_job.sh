#!/usr/bin/env bash
# Ping a Claude Code session about a detached job, so the agent that launched it
# waits without watching. tools/launch_job.sh starts one for every job it
# launches from inside a session; run it yourself only to follow a job another
# session launched (a HANDOFF.md resume).
#
#   tools/notify_job.sh <pid> --log FILE [--session ID] [--progress REGEX] [--label TEXT]
#
#   --log       the job's log
#   --session   the session to ping, by session id. Default: the one this runs
#               in ($CLAUDE_CODE_SESSION_ID)
#   --progress  a log line matching this is worth a ping. Default: stage 3's
#               "[k/N] <fold>:" line, once per fold. Stage 2 has none on purpose:
#               a ping per ident is too many.
#   --label     what pings call the job. Default: the command launch_job.sh
#               registered
#
# Each ping reads "[job <pid> <label> · <HH:MM> · <elapsed>] <event> · log <log>",
# the time included because an agent can't see when a message arrived:
#
#   fold 3/5 (1_150) done           a --progress line
#   error in log, job still running  a new Traceback / OOM / Killed line
#   still running                    no other ping for HEARTBEAT (3000 s); pings
#                                    under an hour apart keep the prompt cache warm
#   DONE (exit 0) / FAILED (exit N) / EXITED, status unknown       then it exits
#
# Nothing to arm or re-arm. It detaches, exits when the job ends or the session
# does, and registers itself in .local/jobs/ so tools/agent_loop.sh kills it with
# the batch's jobs. Pings go through tools/send_to_session.sh, 3 tries each; its
# own log is .local/notify/<job pid>.log.

# Run the main checkout's copy: a worktree's tools/ is frozen at its branch point.
_main="$(dirname "$(git -C "$(dirname "$(realpath "$0")")" rev-parse --path-format=absolute --git-common-dir)")/tools/$(basename "$0")"
[ "$(realpath "$0")" = "$(realpath -m "$_main")" ] || [ ! -f "$_main" ] || exec bash "$_main" "$@"

set -uo pipefail
usage="usage: notify_job.sh <pid> --log FILE [--session ID] [--progress REGEX] [--label TEXT]"
pid=${1:?$usage}; shift
log=""; session=${CLAUDE_CODE_SESSION_ID:-}; progress='\[[0-9]+/[0-9]+\] [^ ].*: '; label=""
while [ $# -gt 0 ]; do
  case $1 in
    --log) log=$2 ;;
    --session) session=$2 ;;
    --progress) progress=$2 ;;
    --label) label=$2 ;;
    *) echo "notify_job.sh: unknown argument $1" >&2; exit 2 ;;
  esac
  shift 2
done
[ -n "$log" ] || { echo "$usage" >&2; exit 2; }
[ -n "$session" ] || { echo "notify_job.sh: not inside a Claude Code session; pass --session <id>" >&2; exit 2; }
HEARTBEAT=${HEARTBEAT:-3000}
POLL=${POLL:-30}
ERRORS='Traceback|MemoryError|Killed|Segmentation fault|OutOfMemory|out of memory'

here=$(dirname "$(realpath "$0")")
root=$(dirname "$(git -C "$here" rev-parse --path-format=absolute --git-common-dir)")
jobs_dir="$root/.local/jobs"

session_entry() {
  claude agents --json 2>/dev/null \
    | jq -c --arg s "$session" '[.[] | select(.sessionId == $s and .pid != null)][0] // empty'
}

# In the caller's shell: check, then relaunch detached and return at once.
if [ -z "${NOTIFY_DETACHED:-}" ]; then
  kill -0 "$pid" 2>/dev/null || { echo "notify_job.sh: no process $pid" >&2; exit 1; }
  name=$(session_entry | jq -r '.name // empty')
  [ -n "$name" ] || { echo "notify_job.sh: session $session isn't running" >&2; exit 1; }
  same=$(claude agents --json 2>/dev/null | jq --arg n "$name" '[.[] | select(.name == $n)] | length')
  [ "$same" = 1 ] || echo "notify_job.sh: warning: $same sessions are named '$name'; pings may reach another one" >&2
  # the registered command, interpreter and directories dropped: "02_set/main.py --set medium ..."
  [ -n "$label" ] || label=$(sed -E 's/^.* :: //; s#^[^ ]*python[^ ]* -u ##; s#^(bash|sh) ##; s#^[^ ]*/([^/ ]+/[^/ ]+)#\1#' \
                               "$jobs_dir/$pid" 2>/dev/null | cut -c1-60)
  mkdir -p "$root/.local/notify" "$jobs_dir"
  NOTIFY_DETACHED=1 nohup setsid bash "$0" "$pid" --log "$log" --session "$session" \
    --progress "$progress" --label "$label" > "$root/.local/notify/$pid.log" 2>&1 < /dev/null &
  npid=$!
  disown
  echo "notifier for job $pid → session $name" > "$jobs_dir/$npid"
  echo "notifier $npid: pings session '$name' on each fold, error and the job's end, and after $((HEARTBEAT / 60)) min $(( HEARTBEAT % 60 ))s without another ping. Nothing to arm; wait for its pings."
  exit 0
fi

stamp() { date '+%F %T'; }
count() { local n; n=$(grep -cE "$1" "$log" 2>/dev/null); echo "${n:-0}"; }
started=$(( $(date +%s) - $(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ' || echo 0) ))
spid=$(session_entry | jq -r '.pid // empty')

ping() {
  local e name msg r i
  e=$(( $(date +%s) - started ))
  name=$(session_entry | jq -r '.name // empty')
  [ -n "$name" ] || { echo "$(stamp) session $session is gone; exiting"; exit 0; }
  msg="[job $pid${label:+ $label} · $(date +%H:%M) · $((e / 3600))h$(( e % 3600 / 60 ))m] $1 · log $log"
  last_ping=$(date +%s)
  for i in 1 2 3; do
    if r=$("$here/send_to_session.sh" "$name" "$msg"); then
      echo "$(stamp) sent: $msg"; return
    fi
    echo "$(stamp) try $i failed ($r): $msg"
    sleep 30
  done
}

last_ping=$(date +%s)
units=$(count "$progress")
errors=$(count "$ERRORS")
while true; do
  [ -z "$spid" ] || kill -0 "$spid" 2>/dev/null || { echo "$(stamp) session exited; exiting"; exit 0; }

  c=$(count "$progress")
  if [ "$c" -gt "$units" ]; then
    units=$c
    line=$(grep -E "$progress" "$log" | tail -n 1)
    if [[ $line =~ \[([0-9]+)/([0-9]+)\]\ ([^:]*): ]]; then
      ping "fold ${BASH_REMATCH[1]}/${BASH_REMATCH[2]} (${BASH_REMATCH[3]}) done"
    else
      ping "progress: ${line:0:100}"
    fi
  fi

  c=$(count "$ERRORS")
  if [ "$c" -gt "$errors" ]; then
    errors=$c
    ping "error in log, job still running"
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    sleep 2  # let the last lines land
    status=$(grep -oE '^\[launch_job\] exit [0-9]+' "$log" 2>/dev/null | tail -n 1 | grep -oE '[0-9]+$')
    if [ "$status" = 0 ]; then ping "DONE (exit 0)"
    elif [ -n "$status" ]; then ping "FAILED (exit $status)"
    else ping "EXITED, status unknown"
    fi
    exit 0
  fi

  [ $(( $(date +%s) - last_ping )) -ge "$HEARTBEAT" ] && ping "still running"
  sleep "$POLL"
done
