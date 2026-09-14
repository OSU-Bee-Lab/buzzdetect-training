#!/usr/bin/env bash
# Deliver a message to a running Claude Code session, by name.
#
#   tools/send_to_session.sh <session name> <message>
#
# The CLI can't message a running session, so a throwaway haiku session relays
# it with SendMessage; it arrives as "Another Claude session sent a message".
# Prints the relay's last line (SENT, or FAILED with the reason) and exits 0
# only on SENT. The prompt goes on stdin: --allowedTools takes a variable number
# of values and would swallow a positional prompt after it.

# Run the main checkout's copy: a worktree's tools/ is frozen at its branch point.
_main="$(dirname "$(git -C "$(dirname "$(realpath "$0")")" rev-parse --path-format=absolute --git-common-dir)")/tools/$(basename "$0")"
[ "$(realpath "$0")" = "$(realpath -m "$_main")" ] || [ ! -f "$_main" ] || exec bash "$_main" "$@"

set -uo pipefail
usage="usage: send_to_session.sh <session name> <message>"
name=${1:?$usage}; msg=${2:?$usage}
root=$(dirname "$(git -C "$(dirname "$(realpath "$0")")" rev-parse --path-format=absolute --git-common-dir)")

r=$(printf '%s' "Use the SendMessage tool once, to the session named $name, with exactly this text, then reply SENT or FAILED with the reason:

$msg" | (cd "$root" && claude -p --model haiku --effort low --no-session-persistence \
           --allowedTools SendMessage ListAgents 2>&1) | tail -n 1)
echo "$r"
[[ $r == SENT* ]]
