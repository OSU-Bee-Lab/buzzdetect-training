#!/usr/bin/env bash
# Chain fresh Claude Code sessions through LOOP.md, N experiments per session.
#
#   ./tools/agent_loop.sh [--experiments 4] [--model sonnet] [--effort medium]
#                         [--fix-model opus] [--fix-effort medium]
#
#   --experiments   experiments per agent
#   --model/--effort          for agents running experiments
#   --fix-model/--fix-effort  for agents fixing what earlier agents reported
#   Sessions run in auto mode; a model without it (haiku) prompts for permissions.
#
# Runs in the foreground, one line per decision (also in .local/agent_loop/driver.log).
#   Ctrl+C          ask the running agent to wrap up; the loop exits once it has
#   Ctrl+C again    stop the agent and kill the jobs it started, now
#   tools/loop_signal.sh stop    exit after the current batch
#
# Each session is a background interactive session (`claude --bg`) with Remote
# Control on, so it can be watched and steered from claude.ai/code or the app;
# a `claude -p` session can't be. An agent reports with tools/loop_signal.sh,
# not by ending its turn -- it also ends turns while waiting on a Monitor:
#   done      batch finished       -> next session gets tools/loop_prompt.md
#   issue     batch hit a blocker  -> next session gets tools/loop_fix_prompt.md
#   friction  reported mid-batch   -> after an experiment batch, a fixer runs
#                                     before the next one (not after a fix batch)
# What a fixer gets is collected into issues/before-batchNNN.md and passed in.
#
# Usage limits: a session that hits one waits and continues at the reset
# (autoContinueAtUsageLimit, forced on) only if the reset is within
# LIMIT_WAIT_MAX (50 min), i.e. while its prompt cache is still warm. A later
# reset, or one the loop can't read off the session's screen, stops the session
# and halts the loop: resuming would re-read the whole context uncached.
#
# The loop also halts when a fix session reports an issue too, or a session
# exits without signalling. Rerunning after a halt resumes: a leftover issue
# gets one more fix attempt; delete .local/agent_loop/issue to skip that.
#
# "The jobs it started" are those tools/launch_job.sh registered in .local/jobs/
# since this loop began; a job started any other way is not killed. A halt
# leaves jobs running.
#
# Env, for testing the loop: POLL (seconds, 60), PROMPT / FIX_PROMPT (templates),
# BLOCKED_GRACE (seconds a session stays blocked before it's flagged, 300).
set -uo pipefail

N=4; MODEL=sonnet; EFFORT=medium; FIX_MODEL=opus; FIX_EFFORT=medium
while [ $# -gt 0 ]; do
  case $1 in
    --experiments) N=$2 ;;
    --model) MODEL=$2 ;;
    --effort) EFFORT=$2 ;;
    --fix-model) FIX_MODEL=$2 ;;
    --fix-effort) FIX_EFFORT=$2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "agent_loop.sh: unknown argument $1 (see --help)" >&2; exit 2 ;;
  esac
  shift 2
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE="$ROOT/.local/agent_loop"
JOBS="$ROOT/.local/jobs"
POLL=${POLL:-60}
PROMPT=${PROMPT:-$ROOT/tools/loop_prompt.md}
FIX_PROMPT=${FIX_PROMPT:-$ROOT/tools/loop_fix_prompt.md}
LIMIT_WAIT_MAX=3000   # ~50 min: the prompt cache's ~1 h TTL with margin
BLOCKED_GRACE=${BLOCKED_GRACE:-300}   # seconds a session stays blocked before it's flagged
SESSION_SETTINGS='{"autoContinueAtUsageLimit": true, "worktree": {"bgIsolation": "none"}}'
OUT="$STATE/.out"
mkdir -p "$STATE/issues" "$JOBS"

log() { echo "$(date '+%F %T') $*" | tee -a "$STATE/driver.log"; }
finish() { rm -f "$STATE/driver.pid" "$OUT"; exit "$1"; }
halt() { log "$* → halting loop"; finish 1; }

# Run a command into $OUT without blocking signals. Bash runs a trap only once
# its foreground command returns, and folds a repeated signal that arrives in
# the meantime into one -- so a double Ctrl+C during a slow `claude` call would
# read as a single one. `wait` on a background job returns as soon as a trap
# fires, then resumes.
quiet() {
  "$@" > "$OUT" 2>/dev/null &
  local p=$!
  while ! wait "$p"; do kill -0 "$p" 2>/dev/null || break; done
}
nap() { quiet sleep "$1"; }

# screen text on stdin -> nothing if no usage limit shows; otherwise the reset
# as epoch seconds, or "unknown" when a limit shows but no reset time parses.
# The screen is raw terminal output, where spaces are often cursor moves.
limit_reset() {
  local text at t now
  text=$(sed -E 's/\x1b\[[0-9]*[CG]/ /g; s/\x1b\[[0-9;?]*[A-Za-z]//g; s/\x1b\][^\x07]*\x07//g' | tr -s ' ')
  grep -aqE "(hit your [A-Za-z]+ limit|Usage limit reached)" <<<"$text" || return 0
  at=$(grep -aoE "(continuing automatically at|resets) [A-Za-z0-9:, ]*[0-9](am|pm)?" <<<"$text" \
       | tail -n 1 | sed -E 's/^(continuing automatically at|resets) //; s/,//g')
  t=$(date -d "$at" +%s 2>/dev/null) || { echo unknown; return 0; }
  now=$(date +%s)
  # a bare clock time earlier than now more than an hour ago means tomorrow
  [ -n "$at" ] && [ "$t" -lt $(( now - 3600 )) ] && t=$(( t + 86400 ))
  [ -n "$at" ] && echo "$t" || echo unknown
}

pid=$(cat "$STATE/driver.pid" 2>/dev/null)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
  echo "another loop is already running (pid $pid); not starting a second one" >&2
  exit 1
fi
echo $$ > "$STATE/driver.pid"
started=$(date +%s)

kill_jobs() {  # every launch_job.sh job registered since this loop started
  local f pid groups=()
  for f in "$JOBS"/*; do
    [ -e "$f" ] || continue
    pid=${f##*/}
    [ "$(stat -c %Y "$f")" -ge "$started" ] || continue
    if [ "$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$pid" ]; then
      kill -TERM -- "-$pid" 2>/dev/null && groups+=("$pid") && log "killing job $pid: $(cat "$f")"
    fi
    rm -f "$f"
  done
  [ ${#groups[@]} -gt 0 ] || { log "no running jobs from this loop to kill"; return; }
  for _ in $(seq 10); do
    local alive=0
    for pid in "${groups[@]}"; do kill -0 -- "-$pid" 2>/dev/null && alive=1; done
    [ $alive = 0 ] && return
    sleep 1
  done
  for pid in "${groups[@]}"; do kill -KILL -- "-$pid" 2>/dev/null && log "job $pid ignored SIGTERM → sent SIGKILL"; done
}

id=""; wrapping_up=0
on_interrupt() {
  if [ -z "$id" ]; then
    log "Ctrl+C with no agent running → exiting loop"
    finish 130
  fi
  if [ "$wrapping_up" = 0 ]; then
    wrapping_up=1
    log "Ctrl+C → asking session $id (loop-$batch) to wrap up; the loop exits when it has. Ctrl+C again to stop it and kill its jobs now"
    # the prompt goes on stdin: --allowedTools takes a variable number of
    # values and would swallow a positional prompt after it
    ( r=$(cd "$ROOT" && printf '%s' "Use the SendMessage tool once, to the session named loop-$batch, with exactly this text, then reply SENT or FAILED with the reason:

Luke pressed Ctrl+C on tools/agent_loop.sh: wrap up now. Start nothing new. If a job is still running, don't wait for it: record the experiment's state and how to resume in its notes.md and a HANDOFF.md (LOOP.md step 3), then commit and push. Otherwise finish recording what is done (LOOP.md step 5). Then run $ROOT/tools/loop_signal.sh done \"wrapped up\" and end your turn; the loop exits when you do." \
          | claude -p --model haiku --effort low --no-session-persistence \
              --allowedTools SendMessage ListAgents 2>&1 | tail -n 1)
      kill -0 $$ 2>/dev/null || exit 0   # the loop already exited (a second Ctrl+C)
      if [[ $r == SENT* ]]; then
        log "wrap-up message delivered to loop-$batch → waiting for it to signal done"
      else
        log "wrap-up message to loop-$batch failed ($r) → it won't wrap up on its own; Ctrl+C again to stop it and kill its jobs"
      fi ) &
    return
  fi
  trap '' INT TERM
  log "Ctrl+C again → stopping session $id and killing the jobs it started"
  claude stop "$id" >/dev/null 2>&1
  kill_jobs
  log "session $id stopped → exiting loop"
  finish 130
}
trap on_interrupt INT TERM

session_field() {  # id, jq filter -> that field of the session's `claude agents` entry
  quiet claude agents --json --all
  jq -r --arg id "$1" ".[] | select(.id == \$id) | $2 // empty" "$OUT" 2>/dev/null
}

# Unfinished experiments to resume: a worktree HANDOFF.md from this era (no
# older than log.jsonl's first entry) whose slug has no log.jsonl entry yet.
# Logging the experiment retires its handoff.
find_handoffs() {
  local era f slug
  era=$(head -n 1 "$ROOT/log.jsonl" 2>/dev/null | jq -r '.date // empty' 2>/dev/null)
  for f in "$ROOT"/.local/worktrees/*/HANDOFF.md; do
    [ -e "$f" ] || continue
    slug=$(basename "$(dirname "$f")")
    [ -n "$era" ] && [[ $(date -r "$f" +%F) < $era ]] && continue
    grep -q "\"name\": \"$slug\"" "$ROOT/log.jsonl" 2>/dev/null && continue
    echo "$f"
  done
}

render() {  # template, issue path
  sed -e "s|{N}|$N|g" -e "s|{BATCH}|$batch|g" -e "s|{ISSUE}|$2|g" -e "s|{ROOT}|$ROOT|g" "$1"
}

log "loop started: $N experiments per agent on $MODEL/$EFFORT, fixes on $FIX_MODEL/$FIX_EFFORT (Ctrl+C to wrap up, twice to kill)"
last_was_fix=0
while true; do
  id=""
  if [ -f "$STATE/stop" ]; then
    rm -f "$STATE/stop"
    log "stop signalled → exiting loop"
    finish 0
  fi

  batch=$(( $(cat "$STATE/batch" 2>/dev/null || echo 0) + 1 ))
  echo "$batch" > "$STATE/batch"

  # A fixer runs for a blocking issue, or for friction reports after an
  # experiment batch. Friction a fixer reports itself waits for the next
  # experiment batch, so fixers can't chain.
  friction=0; [ -s "$STATE/friction.md" ] && friction=$(grep -c '^- ' "$STATE/friction.md")
  if [ -f "$STATE/issue" ] || { [ "$friction" -gt 0 ] && [ "$last_was_fix" = 0 ]; }; then
    [ -f "$STATE/issue" ] && [ "$last_was_fix" = 1 ] \
      && halt "batch $batch: the fix agent reported an issue too (read $STATE/issue, resolve it, rerun)"
    issue="$STATE/issues/before-batch$(printf %03d "$batch").md"
    : > "$issue"; causes=()
    if [ -f "$STATE/issue" ]; then
      { echo "# Blocking issue"; echo; cat "$STATE/issue"; echo; } >> "$issue"
      rm "$STATE/issue"; causes+=("previous agent reported an issue")
    fi
    if [ "$friction" -gt 0 ]; then
      { echo "# Friction"; echo; cat "$STATE/friction.md"; } >> "$issue"
      rm "$STATE/friction.md"; causes+=("$friction friction report(s)")
    fi
    prompt=$(render "$FIX_PROMPT" "$issue"); last_was_fix=1
    model=$FIX_MODEL; effort=$FIX_EFFORT
    cause="$(IFS=+; echo "${causes[*]}" | sed 's/+/ and /') ($issue)"; action="launching a fix agent"
  else
    model=$MODEL; effort=$EFFORT
    if [ "$friction" -gt 0 ]; then
      cause="no open issue; the fixer's own $friction friction report(s) wait until after this batch"
    else
      cause="no open issue"
    fi
    prompt=$(render "$PROMPT" ""); last_was_fix=0
    action="launching an agent for $N experiments"
    handoffs=$(find_handoffs)
    if [ -n "$handoffs" ]; then
      prompt+=$'\n\n'"Before starting anything new, resume each unfinished experiment below by following its HANDOFF.md; each counts toward the $N:"$'\n'"$handoffs"
      cause="$cause; unfinished handoff in $(xargs -n 1 dirname <<<"$handoffs" | xargs -n 1 basename | paste -sd, -)"
    fi
  fi
  rm -f "$STATE/done"

  # LOOP.md has its own worktree discipline (setup_worktree.sh, plus deliberate
  # edits in main: IDEAS.md, new embedders, a fixer's repairs), so Claude Code's
  # background-edit guard is off (bgIsolation in SESSION_SETTINGS) and
  # EnterWorktree, which the guard pushes agents into, is blocked. It also asks
  # for confirmation in auto mode, and Remote Control didn't show that dialog
  # (2026-09-14).
  out=$(cd "$ROOT" && claude --bg --remote-control "loop-$batch" -n "loop-$batch" \
        --disallowedTools "EnterWorktree,ExitWorktree" \
        --permission-mode auto --model "$model" --effort "$effort" \
        --settings "$SESSION_SETTINGS" "$prompt" 2>&1)
  id=$(grep -oP 'backgrounded · \K[0-9a-f]+' <<<"$out")
  [ -n "$id" ] || halt "batch $batch: claude --bg failed to start a session ($out)"
  log "batch $batch: $cause → $action on $model/$effort (session $id, named loop-$batch)"

  # A new session takes a few seconds to appear in `claude agents`, so it only
  # counts as gone once it has been seen running.
  seen=0; launched=$(date +%s); waited_for=""; was_waiting=""; blocked_since=""
  while [ ! -f "$STATE/done" ] && [ ! -f "$STATE/issue" ]; do
    quiet claude agents --json --all
    entry=$(jq -c --arg id "$id" '.[] | select(.id == $id)' "$OUT" 2>/dev/null)
    if [ -n "$(jq -r '.pid // empty' <<<"$entry")" ]; then
      seen=1
      # The loop can't answer for you, but it says when a session needs you. A
      # permission prompt is flagged at once. `state: blocked` is Claude Code's
      # job-state label for any turn that ends waiting on you, including a live
      # Remote Control conversation (a false alarm on 2026-09-14), so it's only
      # flagged once it has lasted BLOCKED_GRACE, with the session's own reason.
      waiting=$(jq -r 'if .status == "waiting" then (.waitingFor // "input needed") else "" end' <<<"$entry")
      if [ -z "$waiting" ] && [ "$(jq -r '.state // empty' <<<"$entry")" = blocked ]; then
        blocked_since=${blocked_since:-$(date +%s)}
        if [ $(( $(date +%s) - blocked_since )) -ge "$BLOCKED_GRACE" ]; then
          detail=$(jq -r '.detail // empty' "$HOME/.claude/jobs/$id/state.json" 2>/dev/null | head -c 300)
          waiting="blocked: ${detail:-no reason given}"
        fi
      else
        blocked_since=""
      fi
      # one line per episode: a blocked session's detail can change while it waits
      if [ -n "$waiting" ] && [ "${waiting%%:*}" != "${was_waiting%%:*}" ]; then
        log "batch $batch: session $id is waiting on you ($waiting) → answer it in loop-$batch; the loop keeps waiting"
      fi
      was_waiting=$waiting
      quiet claude logs "$id"
      reset=$(limit_reset < "$OUT")
      if [ "$reset" = unknown ]; then
        claude stop "$id" >/dev/null 2>&1
        halt "batch $batch: session $id hit a usage limit and its reset time couldn't be read → stopped it (jobs left running)"
      elif [ -n "$reset" ] && [ "$reset" != "$waited_for" ]; then
        wait_s=$(( reset - $(date +%s) ))
        if [ "$wait_s" -gt "$LIMIT_WAIT_MAX" ]; then
          claude stop "$id" >/dev/null 2>&1
          halt "batch $batch: session $id hit a usage limit resetting at $(date -d "@$reset" '+%F %H:%M') ($(( wait_s / 60 )) min, past the cache window) → stopped it (jobs left running)"
        fi
        waited_for=$reset
        log "batch $batch: session $id hit a usage limit resetting at $(date -d "@$reset" +%H:%M) ($(( wait_s > 0 ? wait_s / 60 : 0 )) min, inside the cache window) → letting it continue automatically"
      fi
    elif [ "$seen" = 1 ]; then
      halt "batch $batch: session $id exited without signalling done or issue"
    elif [ $(( $(date +%s) - launched )) -gt 120 ]; then
      halt "batch $batch: session $id never appeared in claude agents within 2 min"
    fi
    nap "$POLL"
  done
  if [ "$wrapping_up" = 1 ]; then
    log "batch $batch: agent finished wrapping up → stopping session $id"
  elif [ -f "$STATE/done" ]; then
    log "batch $batch: agent marked its batch finished → stopping session $id, starting the next batch"
  else
    log "batch $batch: agent reported an issue → stopping session $id, the next agent will try to fix it"
  fi

  # Let the session finish the turn it signalled in, then stop it. It stays
  # resumable (`claude attach`/`--resume`) for reading back.
  for _ in $(seq 20); do
    [ "$(session_field "$id" .status)" = busy ] || break
    nap 30
  done
  claude stop "$id" >/dev/null 2>&1
  if [ "$wrapping_up" = 1 ]; then
    log "wrap-up complete → exiting loop"
    finish 0
  fi
done
