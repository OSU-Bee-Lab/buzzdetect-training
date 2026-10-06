#!/usr/bin/env bash
# LOOP.md steps 5-6, once notes.md is written: commit and push the experiment
# branch, then record it in its arm's log in main and commit that.
#
# Training arm (default): appends the 03_train/log.jsonl entry.
#   tools/finish_experiment.sh <slug> --summary "<one line, used in both commits>" \
#     --model <exp model dir> --baseline-model <matched control> \
#     --hypothesis "..." --trust clean|caveated|artifact --conclusion "..." \
#     [--baseline-name <label, default: control dir name>] [--commit-also IDEAS.md]
#
# Distillation arm: main.py's record stage already wrote one 05_distill/log.jsonl
# row per student; this stamps them with "exp": "<slug>" and commits the log.
#   tools/finish_experiment.sh <slug> --arm distill --summary "..." \
#     --runs "<student name> [<student name> ...]" [--commit-also IDEAS.md]
#
# Model paths are relative to main's root (e.g.
# .local/worktrees/<slug>/models/<name>). The order is the point: the entry is
# validated before anything is committed, the branch is pushed before the log
# names it, and main_commit is main's HEAD at logging time. In main it commits
# only the arm's log and any --commit-also paths, never anything else that is
# dirty there. DRY_RUN=1 prints the git and write steps instead of running them.
# Run the main checkout's copy: a worktree's tools/ is frozen at its branch point.
_main="$(dirname "$(git -C "$(dirname "$(realpath "$0")")" rev-parse --path-format=absolute --git-common-dir)")/tools/$(basename "$0")"
[ ! -f "$_main" ] || [ "$(realpath "$0")" = "$(realpath "$_main")" ] || exec bash "$_main" "$@"

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/tools/python_path.sh"  # sets PY
slug=${1:?usage: finish_experiment.sh <slug> --summary ... --model ... --hypothesis ... --trust ... --conclusion ...}
shift
WT="$ROOT/.local/worktrees/$slug"

summary=""; also=(); entry=(); arm=train; runs=""
while [ $# -gt 0 ]; do
  case $1 in
    --summary) summary=$2 ;;
    --arm) arm=$2 ;;
    --runs) runs=$2 ;;
    --commit-also) also+=("$2") ;;
    # A model path given relative to the caller's cwd (e.g. a worktree's
    # models/<name>) must survive the cd to main below; bare names resolve there.
    --model|--baseline-model) [ -e "$2" ] && entry+=("$1" "$(cd "$2" && pwd)") || entry+=("$1" "$2") ;;
    --baseline-name|--hypothesis|--trust|--conclusion) entry+=("$1" "$2") ;;
    *) echo "finish_experiment.sh: unknown argument $1" >&2; exit 2 ;;
  esac
  shift 2
done

fail() { echo "finish_experiment.sh: $*" >&2; exit 1; }
[ -n "$summary" ] || fail "--summary is required"
case $arm in
  train) LOG=03_train/log.jsonl; [ -z "$runs" ] || fail "--runs is for --arm distill" ;;
  distill) LOG=05_distill/log.jsonl; [ -n "$runs" ] || fail "--arm distill needs --runs"
           [ ${#entry[@]} -eq 0 ] || fail "--arm distill takes no --model/--hypothesis/...: the rows are main.py's" ;;
  *) fail "--arm is train or distill, not $arm" ;;
esac
[ -d "$WT" ] || fail "no worktree at $WT"
[ -f "$WT/notes.md" ] || fail "$WT/notes.md is missing; write it first (LOOP.md step 5)"
! grep -q -e "\"name\": \"$slug\"" -e "\"exp\": \"$slug\"" "$ROOT/03_train/log.jsonl" "$ROOT/05_distill/log.jsonl" \
  || fail "$slug is already logged (03_train/log.jsonl or 05_distill/log.jsonl)"
handoffs=("$WT"/HANDOFF*.md)
[ -e "${handoffs[0]}" ] && fail "$WT still has ${handoffs[*]##*/}; the job it describes is over once notes.md is written -- rm it before finishing" || true
for p in "${also[@]}"; do
  ! git -C "$ROOT" diff --quiet -- "$p" || fail "--commit-also $p: main's copy is unchanged; edit it in main ($ROOT/$p), not the worktree"
done

run() { if [ "${DRY_RUN:-0}" = 1 ]; then printf '+'; printf ' %q' "$@"; echo; else "$@"; fi; }

cd "$ROOT"
read -ra run_names <<< "$runs"   # --runs is one space-separated string
if [ "$arm" = train ]; then
  "$PY" tools/log_entry.py --name "$slug" "${entry[@]}" > /dev/null   # fails here, before any commit
else
  "$PY" 05_distill/log_exp.py --exp "$slug" --runs "${run_names[@]}" --check
fi

run git -C "$WT" add -A
git -C "$WT" diff --cached --quiet || run git -C "$WT" commit -q -m "exp/$slug: $summary"
run git -C "$WT" push -q origin "exp/$slug"

if [ "$arm" = train ]; then
  run "$PY" tools/log_entry.py --name "$slug" "${entry[@]}" --write
else
  run "$PY" 05_distill/log_exp.py --exp "$slug" --runs "${run_names[@]}"
fi
run git add -- "$LOG" "${also[@]}"
run git commit -q -m "log: $slug -- $summary" -- "$LOG" "${also[@]}"
[ "${DRY_RUN:-0}" = 1 ] && { echo "dry run: nothing committed"; exit 0; }
echo "finished $slug: exp/$slug pushed, $LOG committed in main ($(git rev-parse --short HEAD))"
