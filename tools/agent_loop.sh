#!/usr/bin/env bash
# Chain fresh Claude Code sessions through LOOP.md, N experiments per session.
#
#   ./tools/agent_loop.sh [--experiments 4] [--model sonnet] [--effort medium]
#                         [--fix-model opus] [--fix-effort medium]
#
#   --experiments   experiments per agent
#   --model/--effort          for agents running experiments
#   --fix-model/--fix-effort  for agents fixing an issue the previous one reported
#
# Runs in the foreground, one line per decision (also appended to
# .local/agent_loop/driver.log).
#   Ctrl+C                         end the loop now; the running agent is left
#                                  running, and the line says how to stop it
#   touch .local/agent_loop/stop   end the loop after the current batch
#
# Each session is a background interactive session (`claude --bg`) with Remote
# Control on, so it can be watched and steered from claude.ai/code or the app;
# a `claude -p` session can't be. A session ends its batch by writing a marker
# file, not by ending its turn -- it also ends turns while waiting on a Monitor.
#
# Markers, all under .local/agent_loop/:
#   done   batch finished        -> next session gets tools/loop_prompt.md
#   issue  batch hit a blocker   -> next session gets tools/loop_fix_prompt.md
#                                   (the file is moved to issues/ and passed in)
# The loop halts when a fix session also ends on `issue`, or when a session's
# process exits with no marker. Rerunning after a halt resumes: a leftover
# `issue` gets one more fix attempt; delete it to skip that.
#
# Env, for testing the loop: POLL (seconds, 60), PROMPT / FIX_PROMPT (templates).
set -uo pipefail

N=4; MODEL=sonnet; EFFORT=medium; FIX_MODEL=opus; FIX_EFFORT=medium
while [ $# -gt 0 ]; do
  case $1 in
    --experiments) N=$2 ;;
    --model) MODEL=$2 ;;
    --effort) EFFORT=$2 ;;
    --fix-model) FIX_MODEL=$2 ;;
    --fix-effort) FIX_EFFORT=$2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "agent_loop.sh: unknown argument $1 (see --help)" >&2; exit 2 ;;
  esac
  shift 2
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE="$ROOT/.local/agent_loop"
POLL=${POLL:-60}
PROMPT=${PROMPT:-$ROOT/tools/loop_prompt.md}
FIX_PROMPT=${FIX_PROMPT:-$ROOT/tools/loop_fix_prompt.md}
mkdir -p "$STATE/issues"

log() { echo "$(date '+%F %T') $*" | tee -a "$STATE/driver.log"; }
finish() { rm -f "$STATE/driver.pid"; exit "$1"; }
halt() { log "$* → halting loop"; finish 1; }

pid=$(cat "$STATE/driver.pid" 2>/dev/null)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
  echo "another loop is already running (pid $pid); not starting a second one" >&2
  exit 1
fi
echo $$ > "$STATE/driver.pid"

id=""
on_interrupt() {
  if [ -n "$id" ]; then
    log "interrupted (Ctrl+C) → exiting loop; session $id (loop-$batch) left running, stop it with: claude stop $id"
  else
    log "interrupted (Ctrl+C) → exiting loop"
  fi
  finish 130
}
trap on_interrupt INT TERM

session() {  # id -> the session's entry in `claude agents`, empty if none
  claude agents --json --all 2>/dev/null | jq -c --arg id "$1" '.[] | select(.id == $id)'
}

render() {  # template, issue path
  sed -e "s|{N}|$N|g" -e "s|{BATCH}|$batch|g" -e "s|{ISSUE}|$2|g" "$1"
}

log "loop started: $N experiments per agent on $MODEL/$EFFORT, fixes on $FIX_MODEL/$FIX_EFFORT (Ctrl+C to end now; touch $STATE/stop to end after this batch)"
last_was_fix=0
while true; do
  id=""
  if [ -f "$STATE/stop" ]; then
    rm -f "$STATE/stop"
    log "stop file found → exiting loop"
    finish 0
  fi

  batch=$(( $(cat "$STATE/batch" 2>/dev/null || echo 0) + 1 ))
  echo "$batch" > "$STATE/batch"

  if [ -f "$STATE/issue" ]; then
    [ "$last_was_fix" = 1 ] && halt "batch $batch: the fix agent reported an issue too (read $STATE/issue, resolve it, rerun)"
    issue="$STATE/issues/before-batch$(printf %03d "$batch").md"
    mv "$STATE/issue" "$issue"
    prompt=$(render "$FIX_PROMPT" "$issue"); last_was_fix=1
    model=$FIX_MODEL; effort=$FIX_EFFORT
    cause="previous agent reported an issue ($issue)"; action="launching a fix agent"
  else
    prompt=$(render "$PROMPT" ""); last_was_fix=0
    model=$MODEL; effort=$EFFORT
    cause="no open issue"; action="launching an agent for $N experiments"
  fi
  rm -f "$STATE/done"

  out=$(cd "$ROOT" && claude --bg --remote-control "loop-$batch" -n "loop-$batch" \
        --permission-mode auto --model "$model" --effort "$effort" "$prompt" 2>&1)
  id=$(grep -oP 'backgrounded · \K[0-9a-f]+' <<<"$out")
  [ -n "$id" ] || halt "batch $batch: claude --bg failed to start a session ($out)"
  log "batch $batch: $cause → $action on $model/$effort (session $id, named loop-$batch)"

  # A new session takes a few seconds to appear in `claude agents`, so it only
  # counts as gone once it has been seen running.
  seen=0; launched=$(date +%s)
  while [ ! -f "$STATE/done" ] && [ ! -f "$STATE/issue" ]; do
    if session "$id" | jq -e '.pid' >/dev/null; then
      seen=1
    elif [ "$seen" = 1 ]; then
      halt "batch $batch: session $id exited without writing done or issue"
    elif [ $(( $(date +%s) - launched )) -gt 120 ]; then
      halt "batch $batch: session $id never appeared in claude agents within 2 min"
    fi
    sleep "$POLL"
  done
  if [ -f "$STATE/done" ]; then
    log "batch $batch: agent marked its batch finished → stopping session $id, starting the next batch"
  else
    log "batch $batch: agent reported an issue → stopping session $id, the next agent will try to fix it"
  fi

  # Let the session finish the turn it wrote the marker in, then stop it. It
  # stays resumable (`claude attach`/`--resume`) for reading back.
  for _ in $(seq 20); do
    [ "$(session "$id" | jq -r '.status // empty')" = busy ] || break
    sleep 30
  done
  claude stop "$id" >/dev/null 2>&1
done
