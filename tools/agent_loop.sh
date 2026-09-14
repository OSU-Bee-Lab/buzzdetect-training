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
# not by ending its turn -- it also ends turns while waiting on a job's pings:
#   done      batch finished       -> next session gets tools/loop_prompt.md
#   issue     batch hit a blocker  -> next session gets tools/loop_fix_prompt.md
#   friction  reported mid-batch   -> after an experiment batch, a fixer runs
#                                     before the next one (not after a fix batch)
# What a fixer gets is collected into issues/before-batchNNN.md and passed in.
#
# Usage limits: a session that hits one waits for the reset only if it's within
# LIMIT_WAIT_MAX (50 min), i.e. while its prompt cache is still warm. A minute
# past the reset the loop tells it to continue: autoContinueAtUsageLimit is on
# but didn't fire in a --bg session. A later reset, or one the loop can't read
# off the session's screen, stops the session and halts the loop: resuming would
# re-read the whole context uncached.
#
# The loop also halts when a fix session reports an issue too, or a session
# exits without signalling. Rerunning after a halt resumes: a leftover issue
# gets one more fix attempt; delete .local/agent_loop/issue to skip that.
#
# Closing the terminal kills only the loop, not its session. Rerunning then
# shows the log's recent history and reattaches to the batch's session
# ("Batch <N> - Experiment" or "Batch <N> - Fixes") if it's still running,
# instead of launching a second agent.
#
# Cleanup is the loop's job, not the agent's. Whenever a batch ends (done, issue
# or wrap-up) the loop stops the session and kills the jobs it started, with
# their notifiers: those tools/launch_job.sh registered in .local/jobs/ since the
# session launched (a reattached session's own start time counts). A job started
# any other way is not killed. Killing costs at most the fold or ident in
# progress; stages 2 and 3 resume on rerun. A halt leaves jobs running, for you
# to look at.
#
# Env, for testing the loop: POLL (seconds, 60), PROMPT / FIX_PROMPT (templates),
# BLOCKED_GRACE (seconds a session stays blocked before it's flagged, 300).
# Run the main checkout's copy: a worktree's tools/ is frozen at its branch point.
_main="$(dirname "$(git -C "$(dirname "$(realpath "$0")")" rev-parse --path-format=absolute --git-common-dir)")/tools/$(basename "$0")"
[ "$(realpath "$0")" = "$(realpath -m "$_main")" ] || [ ! -f "$_main" ] || exec bash "$_main" "$@"

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
LIMIT_NUDGE=60        # seconds past a usage-limit reset before telling the session to continue
LIMIT_GIVEUP=900      # seconds past the reset before a still-blocked session is flagged
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

send_to_session() {  # session name, message -> SENT, or FAILED with the reason
  "$ROOT/tools/send_to_session.sh" "$@"
}

pid=$(cat "$STATE/driver.pid" 2>/dev/null)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
  echo "another loop is already running (pid $pid); not starting a second one" >&2
  exit 1
fi
echo $$ > "$STATE/driver.pid"
started=$(date +%s)

kill_jobs() {  # since (epoch s) -> kill every launch_job.sh job registered since then
  local since=$1 f pid groups=()
  for f in "$JOBS"/*; do
    [ -e "$f" ] || continue
    pid=${f##*/}
    [ "$(stat -c %Y "$f")" -ge "$since" ] || continue
    if [ "$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$pid" ]; then
      kill -TERM -- "-$pid" 2>/dev/null && groups+=("$pid") && log "killing job $pid: $(cat "$f")" \
        && { ! grep -q ' :: ' "$f" \
             || echo "[agent_loop] killed by loop cleanup $(date '+%F %T'); not a crash, rerun resumes" >> "$(sed 's/ :: .*//' "$f")"; }
    fi
    rm -f "$f"
  done
  [ ${#groups[@]} -gt 0 ] || { log "no running jobs from this session to kill"; return; }
  for _ in $(seq 10); do
    local alive=0
    for pid in "${groups[@]}"; do kill -0 -- "-$pid" 2>/dev/null && alive=1; done
    [ $alive = 0 ] && return
    sleep 1
  done
  for pid in "${groups[@]}"; do kill -KILL -- "-$pid" 2>/dev/null && log "job $pid ignored SIGTERM → sent SIGKILL"; done
}

id=""; sname=""; wrapping_up=0; session_started=$started   # sname: "Batch N - Experiment" / "Batch N - Fixes"
on_interrupt() {
  if [ -z "$id" ]; then
    log "Ctrl+C with no agent running → exiting loop"
    finish 130
  fi
  if [ "$wrapping_up" = 0 ]; then
    wrapping_up=1
    log "Ctrl+C → asking session $id ($sname) to wrap up; the loop exits when it has. Ctrl+C again to stop it and kill its jobs now"
    ( r=$(send_to_session "$sname" "Luke pressed Ctrl+C on tools/agent_loop.sh: wrap up now. Start nothing new. Once you signal, the loop stops this session and kills every job you launched, so don't wait for a job and don't stop jobs or their notifiers yourself. If a job is still running, record the experiment's state and its exact relaunch command in its notes.md and a HANDOFF.md (LOOP.md step 3), then commit and push. Otherwise finish recording what is done (LOOP.md step 5). Then run $ROOT/tools/loop_signal.sh done \"wrapped up\" and end your turn.")
      kill -0 $$ 2>/dev/null || exit 0   # the loop already exited (a second Ctrl+C)
      if [[ $r == SENT* ]]; then
        log "wrap-up message delivered to $sname → waiting for it to signal done"
      else
        log "wrap-up message to $sname failed ($r) → it won't wrap up on its own; Ctrl+C again to stop it and kill its jobs"
      fi ) &
    return
  fi
  trap '' INT TERM
  log "Ctrl+C again → stopping session $id and killing the jobs it started"
  claude stop "$id" >/dev/null 2>&1
  kill_jobs "$session_started"
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

# The latest batch's history, from its first log line on
last_batch=$(cat "$STATE/batch" 2>/dev/null)
from=$(grep -n " batch $last_batch:" "$STATE/driver.log" 2>/dev/null | head -n 1 | cut -d: -f1)
if [ -n "$last_batch" ] && [ -n "$from" ]; then
  echo "── batch $last_batch so far ($STATE/driver.log) ──"
  tail -n +"$from" "$STATE/driver.log"
  echo "──"
fi
# The previous loop's session, if it outlived that loop (e.g. a closed terminal).
# Found by name, so it also works for a session launched before this was added.
prev_batch=$(cat "$STATE/batch" 2>/dev/null)
reattach=""
if [ -n "$prev_batch" ]; then
  quiet claude agents --json --all
  # "loop-N" is the name sessions had before 2026-09-14
  found=$(jq -c --arg legacy "loop-$prev_batch" --arg prefix "Batch $prev_batch - " \
    '[.[] | select(.pid != null and (.name == $legacy or ((.name // "") | startswith($prefix))))][0] // empty' \
    "$OUT" 2>/dev/null)
  reattach=$(jq -r '.id // empty' <<<"$found"); reattach_name=$(jq -r '.name // empty' <<<"$found")
fi
last_was_fix=$(cat "$STATE/last_was_fix" 2>/dev/null || echo 0)
log "loop started: $N experiments per agent on $MODEL/$EFFORT, fixes on $FIX_MODEL/$FIX_EFFORT (Ctrl+C to wrap up, twice to kill)"
while true; do
  id=""
  if [ -z "$reattach" ] && [ -f "$STATE/stop" ]; then
    rm -f "$STATE/stop"
    log "stop signalled → exiting loop"
    finish 0
  fi

  if [ -n "$reattach" ]; then
    id=$reattach; batch=$prev_batch; sname=$reattach_name; reattach=""
    st=$(session_field "$id" .startedAt)   # ms
    session_started=${st:+$(( st / 1000 ))}; session_started=${session_started:-$started}
    log "batch $batch: session $id ($sname) is still running → reattaching to it"
  else
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
      prompt=$(render "$FIX_PROMPT" "$issue"); last_was_fix=1; sname="Batch $batch - Fixes"
      model=$FIX_MODEL; effort=$FIX_EFFORT
      cause="$(IFS=+; echo "${causes[*]}" | sed 's/+/ and /') ($issue)"; action="launching a fix agent"
    else
      model=$MODEL; effort=$EFFORT
      if [ "$friction" -gt 0 ]; then
        cause="no open issue; the fixer's own $friction friction report(s) wait until after this batch"
      else
        cause="no open issue"
      fi
      prompt=$(render "$PROMPT" ""); last_was_fix=0; sname="Batch $batch - Experiment"
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
    session_started=$(date +%s)
    out=$(cd "$ROOT" && claude --bg --remote-control "$sname" -n "$sname" \
          --disallowedTools "EnterWorktree,ExitWorktree" \
          --permission-mode auto --model "$model" --effort "$effort" \
          --settings "$SESSION_SETTINGS" "$prompt" 2>&1)
    id=$(grep -oP 'backgrounded · \K[0-9a-f]+' <<<"$out")
    [ -n "$id" ] || halt "batch $batch: claude --bg failed to start a session ($out)"
    log "batch $batch: $cause → $action on $model/$effort (session $id, named $sname)"
    echo "$last_was_fix" > "$STATE/last_was_fix"
  fi

  # A new session takes a few seconds to appear in `claude agents`, so it only
  # counts as gone once it has been seen running.
  seen=0; launched=$(date +%s); was_waiting=""; blocked_since=""
  limit_at=""; limit_phase=""   # the latest usage limit: its reset, and waiting|nudged|resumed
  while [ ! -f "$STATE/done" ] && [ ! -f "$STATE/issue" ]; do
    quiet claude agents --json --all
    entry=$(jq -c --arg id "$id" '.[] | select(.id == $id)' "$OUT" 2>/dev/null)
    if [ -n "$(jq -r '.pid // empty' <<<"$entry")" ]; then
      seen=1
      state=$(jq -r '.state // empty' <<<"$entry")
      now=$(date +%s)

      quiet claude logs "$id"
      reset=$(limit_reset < "$OUT")
      if [ "$reset" = unknown ]; then
        claude stop "$id" >/dev/null 2>&1
        halt "batch $batch: session $id hit a usage limit and its reset time couldn't be read → stopped it (jobs left running)"
      elif [ -n "$reset" ] && [ "$reset" != "$limit_at" ]; then
        wait_s=$(( reset - now ))
        if [ "$wait_s" -gt "$LIMIT_WAIT_MAX" ]; then
          claude stop "$id" >/dev/null 2>&1
          halt "batch $batch: session $id hit a usage limit resetting at $(date -d "@$reset" '+%F %H:%M') ($(( wait_s / 60 )) min, past the cache window) → stopped it (jobs left running)"
        fi
        limit_at=$reset; limit_phase=waiting
        log "batch $batch: session $id hit a usage limit resetting at $(date -d "@$reset" +%H:%M) ($(( wait_s > 0 ? wait_s / 60 : 0 )) min, inside the cache window) → waiting for the reset"
      fi
      # autoContinueAtUsageLimit didn't resume a --bg session on 2026-09-14: a
      # client that opened the session did, 3.5 min after the reset. So once the
      # reset is LIMIT_NUDGE old, the loop tells the session to continue.
      if [ "$limit_phase" = waiting ] || [ "$limit_phase" = nudged ]; then
        if [ "$state" != blocked ] && [ "$now" -ge "$limit_at" ]; then
          limit_phase=resumed
          log "batch $batch: session $id resumed after its usage limit"
        elif [ "$limit_phase" = waiting ] && [ "$now" -ge $(( limit_at + LIMIT_NUDGE )) ]; then
          limit_phase=nudged
          r=$(send_to_session "$sname" "Your usage limit reset at $(date -d "@$limit_at" +%H:%M). Continue from where you left off. Detached jobs kept running through the limit, and so did their notifiers, which keep pinging you: there is nothing to re-arm.")
          if [[ $r == SENT* ]]; then
            log "batch $batch: session $id hasn't resumed since its usage limit reset → told it to continue"
          else
            log "batch $batch: session $id hasn't resumed since its usage limit reset, and messaging it failed ($r) → answer it in $sname"
          fi
        fi
      fi

      # The loop can't answer for you, but it says when a session needs you. A
      # permission prompt is flagged at once. `state: blocked` is Claude Code's
      # job-state label for any turn that ends waiting on you, including a live
      # Remote Control conversation (a false alarm on 2026-09-14), so it's only
      # flagged once it has lasted BLOCKED_GRACE, with the session's own reason.
      # A session waiting out a usage limit is blocked too; that counts only
      # once LIMIT_GIVEUP has passed since the reset.
      waiting=$(jq -r 'if .status == "waiting" then (.waitingFor // "input needed") else "" end' <<<"$entry")
      in_limit=""
      if { [ "$limit_phase" = waiting ] || [ "$limit_phase" = nudged ]; } && [ "$now" -lt $(( limit_at + LIMIT_GIVEUP )) ]; then
        in_limit=1
      fi
      if [ -z "$waiting" ] && [ -z "$in_limit" ] && [ "$state" = blocked ]; then
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
        log "batch $batch: session $id is waiting on you ($waiting) → answer it in $sname; the loop keeps waiting"
      fi
      was_waiting=$waiting
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
    # .state, not .status: status stays busy while any background task runs
    [ "$(session_field "$id" .state)" = working ] || break
    nap 30
  done
  claude stop "$id" >/dev/null 2>&1
  kill_jobs "$session_started"
  if [ "$wrapping_up" = 1 ]; then
    log "wrap-up complete → exiting loop"
    finish 0
  fi
done
