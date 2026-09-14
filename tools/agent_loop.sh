#!/usr/bin/env bash
# Chain fresh Claude Code sessions through LOOP.md, N experiments per session.
#
# Each session is a background interactive session (`claude --bg`) with Remote
# Control on, so it can be watched and steered from claude.ai/code or the app;
# a `claude -p` session can't be. A session ends its batch by writing a marker
# file, not by ending its turn -- it also ends turns while waiting on a Monitor.
#
#   nohup bash tools/agent_loop.sh >/dev/null 2>&1 &   # log: .local/agent_loop/driver.log
#   touch .local/agent_loop/stop                        # exit after the current batch
#
# Markers, all under .local/agent_loop/:
#   done   batch finished        -> next session gets tools/loop_prompt.md
#   issue  batch hit a blocker   -> next session gets tools/loop_fix_prompt.md
#                                   (the file is moved to issues/ and passed in)
# The driver halts, for Luke, when a fix session also ends on `issue`, or when a
# session's process exits with no marker. Rerunning after a halt resumes: a
# leftover `issue` gets one more fix attempt; delete it to skip that.
#
# Env: N (experiments per batch, default 4), POLL (seconds, 60), MODEL, EFFORT,
#      PROMPT / FIX_PROMPT (template paths; override to test the driver).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE="$ROOT/.local/agent_loop"
N=${N:-4}
POLL=${POLL:-60}
PROMPT=${PROMPT:-$ROOT/tools/loop_prompt.md}
FIX_PROMPT=${FIX_PROMPT:-$ROOT/tools/loop_fix_prompt.md}
mkdir -p "$STATE/issues"

log() { echo "$(date '+%F %T') $*" | tee -a "$STATE/driver.log"; }

session() {  # id -> the session's entry in `claude agents`, empty if none
  claude agents --json --all 2>/dev/null | jq -c --arg id "$1" '.[] | select(.id == $id)'
}

render() {  # template, issue path
  sed -e "s|{N}|$N|g" -e "s|{BATCH}|$batch|g" -e "s|{ISSUE}|$2|g" "$1"
}

last_was_fix=0
while true; do
  if [ -f "$STATE/stop" ]; then
    rm -f "$STATE/stop"; log "stop file found; exiting"; exit 0
  fi

  batch=$(( $(cat "$STATE/batch" 2>/dev/null || echo 0) + 1 ))
  echo "$batch" > "$STATE/batch"

  if [ -f "$STATE/issue" ]; then
    if [ "$last_was_fix" = 1 ]; then
      log "fix session ended on an issue too; halting for Luke: $STATE/issue"; exit 1
    fi
    issue="$STATE/issues/before-batch$(printf %03d "$batch").md"
    mv "$STATE/issue" "$issue"
    prompt=$(render "$FIX_PROMPT" "$issue"); last_was_fix=1
  else
    prompt=$(render "$PROMPT" ""); last_was_fix=0
  fi
  rm -f "$STATE/done"

  out=$(cd "$ROOT" && claude --bg --remote-control "loop-$batch" -n "loop-$batch" \
        --permission-mode auto ${MODEL:+--model "$MODEL"} ${EFFORT:+--effort "$EFFORT"} \
        "$prompt" 2>&1)
  id=$(grep -oP 'backgrounded · \K[0-9a-f]+' <<<"$out")
  [ -n "$id" ] || { log "batch $batch: launch failed: $out"; exit 1; }
  log "batch $batch: session $id ($([ $last_was_fix = 1 ] && echo "fix: $issue" || echo "$N experiments"))"

  # A new session takes a few seconds to appear in `claude agents`, so it only
  # counts as gone once it has been seen running.
  seen=0; launched=$(date +%s)
  while [ ! -f "$STATE/done" ] && [ ! -f "$STATE/issue" ]; do
    if session "$id" | jq -e '.pid' >/dev/null; then
      seen=1
    elif [ "$seen" = 1 ]; then
      log "batch $batch: session $id exited without a marker; halting"; exit 1
    elif [ $(( $(date +%s) - launched )) -gt 120 ]; then
      log "batch $batch: session $id never appeared in claude agents; halting"; exit 1
    fi
    sleep "$POLL"
  done
  log "batch $batch: $([ -f "$STATE/done" ] && echo done || echo issue)"

  # Let the session finish the turn it wrote the marker in, then stop it. It
  # stays resumable (`claude attach`/`--resume`) for reading back.
  for _ in $(seq 20); do
    [ "$(session "$id" | jq -r '.status // empty')" = busy ] || break
    sleep 30
  done
  claude stop "$id" >/dev/null 2>&1
done
