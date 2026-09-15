#!/usr/bin/env bash
# How an agent run by tools/agent_loop.sh tells the loop where it stands. Works
# from any worktree: it always writes to the main checkout's .local/agent_loop/.
#
#   tools/loop_signal.sh done "<slugs, or what was fixed>"
#   tools/loop_signal.sh issue <<'EOF'      (blocked; a fixer could repair it)
#   what broke, where it's recorded, what you tried, the experiment's state
#   EOF
#   tools/loop_signal.sh halt <<'EOF'
#   what only Luke can resolve, where it's recorded, the experiment's state
#   EOF
#   tools/loop_signal.sh friction "<what slowed you, how you got past it, commits>"
#   tools/loop_signal.sh stop     the loop exits after this batch
#
# `done`, `issue` and `halt` end your session (a fixer's `done` hands the batch
# to its experiment agent; `halt` also quits the loop, with no fixer and your
# jobs left running): send one as the last thing before ending
# your turn, and the loop stops the session soon after. `friction` doesn't: it
# appends to a list a fixer agent works through at the start of the next batch,
# and you carry on.
# Run the main checkout's copy: a worktree's tools/ is frozen at its branch point.
_main="$(dirname "$(git -C "$(dirname "$(realpath "$0")")" rev-parse --path-format=absolute --git-common-dir)")/tools/$(basename "$0")"
[ "$(realpath "$0")" = "$(realpath -m "$_main")" ] || [ ! -f "$_main" ] || exec bash "$_main" "$@"

set -euo pipefail

usage="usage: loop_signal.sh done <summary> | issue [details, or on stdin] | halt [details, or on stdin] | friction <what happened> | stop"
signal=${1:?$usage}; shift

common=$(git -C "$(dirname "$(realpath "$0")")" rev-parse --path-format=absolute --git-common-dir)
state="$(dirname "$common")/.local/agent_loop"
mkdir -p "$state"

case $signal in
  done)
    [ $# -gt 0 ] || { echo "$usage" >&2; exit 2; }
    echo "$*" > "$state/done" ;;
  issue)
    # appended, so a second report doesn't erase the first
    if [ $# -gt 0 ]; then details="$*"; else details=$(cat); fi
    [ -n "$details" ] || { echo "an issue needs details" >&2; exit 2; }
    echo "$details" >> "$state/issue" ;;
  halt)
    if [ $# -gt 0 ]; then echo "$*" > "$state/halt"; else cat > "$state/halt"; fi
    [ -s "$state/halt" ] || { rm -f "$state/halt"; echo "a halt needs details" >&2; exit 2; } ;;
  friction)
    [ $# -gt 0 ] || { echo "$usage" >&2; exit 2; }
    echo "- $(date '+%F %H:%M') (from $(basename "$PWD")): $*" >> "$state/friction.md" ;;
  stop)
    touch "$state/stop" ;;
  *)
    echo "$usage" >&2; exit 2 ;;
esac
echo "signalled $signal to the loop ($state)"
