#!/usr/bin/env bash
# LOOP.md steps 5-6, once notes.md is written: commit and push the experiment
# branch, then append its log.jsonl entry in main and commit that.
#
#   tools/finish_experiment.sh <slug> --summary "<one line, used in both commits>" \
#     --model <exp model dir> --baseline-model <matched control> \
#     --hypothesis "..." --trust clean|caveated|artifact --conclusion "..." \
#     [--baseline-name <label>] [--commit-also IDEAS.md]
#
# Model paths are relative to main's root (e.g.
# .local/worktrees/<slug>/models/<name>). The order is the point: the entry is
# validated before anything is committed, the branch is pushed before the log
# names it, and main_commit is main's HEAD at logging time. In main it commits
# only log.jsonl and any --commit-also paths, never anything else that is
# dirty there. DRY_RUN=1 prints the git and write steps instead of running them.
# Run the main checkout's copy: a worktree's tools/ is frozen at its branch point.
_main="$(dirname "$(git -C "$(dirname "$(realpath "$0")")" rev-parse --path-format=absolute --git-common-dir)")/tools/$(basename "$0")"
[ "$(realpath "$0")" = "$(realpath -m "$_main")" ] || [ ! -f "$_main" ] || exec bash "$_main" "$@"

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/home/luke/anaconda3/envs/buzzdetect-train/bin/python
slug=${1:?usage: finish_experiment.sh <slug> --summary ... --model ... --hypothesis ... --trust ... --conclusion ...}
shift
WT="$ROOT/.local/worktrees/$slug"

summary=""; also=(); entry=()
while [ $# -gt 0 ]; do
  case $1 in
    --summary) summary=$2 ;;
    --commit-also) also+=("$2") ;;
    --model|--baseline-model|--baseline-name|--hypothesis|--trust|--conclusion) entry+=("$1" "$2") ;;
    *) echo "finish_experiment.sh: unknown argument $1" >&2; exit 2 ;;
  esac
  shift 2
done

fail() { echo "finish_experiment.sh: $*" >&2; exit 1; }
[ -n "$summary" ] || fail "--summary is required"
[ -d "$WT" ] || fail "no worktree at $WT"
[ -f "$WT/notes.md" ] || fail "$WT/notes.md is missing; write it first (LOOP.md step 5)"
! grep -q "\"name\": \"$slug\"" "$ROOT/log.jsonl" || fail "log.jsonl already has an entry named $slug"
for p in "${also[@]}"; do
  ! git -C "$ROOT" diff --quiet -- "$p" || fail "--commit-also $p: main's copy is unchanged; edit it in main ($ROOT/$p), not the worktree"
done

run() { if [ "${DRY_RUN:-0}" = 1 ]; then printf '+'; printf ' %q' "$@"; echo; else "$@"; fi; }

cd "$ROOT"
"$PY" tools/log_entry.py --name "$slug" "${entry[@]}" > /dev/null   # fails here, before any commit

run git -C "$WT" add -A
git -C "$WT" diff --cached --quiet || run git -C "$WT" commit -q -m "exp/$slug: $summary"
run git -C "$WT" push -q origin "exp/$slug"

run "$PY" tools/log_entry.py --name "$slug" "${entry[@]}" --write
run git add -- log.jsonl "${also[@]}"
run git commit -q -m "log: $slug -- $summary" -- log.jsonl "${also[@]}"
[ "${DRY_RUN:-0}" = 1 ] && { echo "dry run: nothing committed"; exit 0; }
echo "finished $slug: exp/$slug pushed, log.jsonl committed in main ($(git rev-parse --short HEAD))"
