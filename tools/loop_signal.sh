#!/usr/bin/env bash
# How an agent run by tools/agent_loop.sh tells the loop where it stands. Works
# from any worktree: it always writes to the main checkout's .local/agent_loop/.
#
#   tools/loop_signal.sh done "<slugs, or what was fixed>"
#   tools/loop_signal.sh issue <<'EOF'
#   what broke, where it's recorded, what you tried, the experiment's state
#   EOF
#   tools/loop_signal.sh friction "<clunky or broken tooling/harness, and what happened>"
#   tools/loop_signal.sh stop     the loop exits after this batch
#
# `done` and `issue` end the batch: send one as the last thing before ending
# your turn, and the loop stops the session soon after. `friction` doesn't: it
# appends to a list a fixer agent works through after the batch, and you carry on.
set -euo pipefail

usage="usage: loop_signal.sh done <summary> | issue [details, or on stdin] | friction <what happened> | stop"
signal=${1:?$usage}; shift

common=$(git -C "$(dirname "$(realpath "$0")")" rev-parse --path-format=absolute --git-common-dir)
state="$(dirname "$common")/.local/agent_loop"
mkdir -p "$state"

case $signal in
  done)
    [ $# -gt 0 ] || { echo "$usage" >&2; exit 2; }
    echo "$*" > "$state/done" ;;
  issue)
    if [ $# -gt 0 ]; then echo "$*" > "$state/issue"; else cat > "$state/issue"; fi
    [ -s "$state/issue" ] || { rm -f "$state/issue"; echo "an issue needs details" >&2; exit 2; } ;;
  friction)
    [ $# -gt 0 ] || { echo "$usage" >&2; exit 2; }
    echo "- $(date '+%F %H:%M') (from $(basename "$PWD")): $*" >> "$state/friction.md" ;;
  stop)
    touch "$state/stop" ;;
  *)
    echo "$usage" >&2; exit 2 ;;
esac
echo "signalled $signal to the loop ($state)"
