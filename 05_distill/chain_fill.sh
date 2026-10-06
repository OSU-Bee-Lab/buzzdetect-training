#!/bin/bash
# Fill out the rung-C frontier for buzz+rain+human students, and separate data from steps (2026-10-04).
# Steps run in order on the one GPU; a failed step is retried once (main.py resumes), then the chain moves on.
# Re-running resumes it. Every row lands in the tracked log, 05_distill/log.jsonl.
#   1 fast32h16 a0.50 on rung B, WSD 7k-56k, no stop: against the rung-C curve, is C's gain data or steps?
#   2 yamnet a0.25 rung C, stop rule: the frontier near the 1.5x speed floor (only a rung-B point there now)
#   3 rung-C mels for fast32 + twofast32 (one decode), then fast32 a0.50 rung C, stop rule (2.2x)
#   4 twofast32 a0.50 rung C, stop rule (1.8x)
#   5 repeat rule: fast32h16 a0.50 rung C at seed 2 to 28k (noise on the best clean frontier point)
#   6 proxy table, wsd curves, plots
#
#   tools/launch_job.sh 05_distill/data/chain_fill.log -- bash 05_distill/chain_fill.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source tools/python_path.sh   # PY
export PYTHONUNBUFFERED=1
export DISTILL_TEACHER=v4-ft-ps-e60-moderate
BRH=classes=ins_buzz+ambient_rain+human
FAILED=()

attempt() {  # attempt <label> <cmd...>: run, retry once
  local label=$1; shift
  for n in 1 2; do
    echo "[fillchain] $label: attempt $n start $(date +%T)"
    if "$@"; then echo "[fillchain] $label: done $(date +%T)"; return 0; fi
    echo "[fillchain] $label: attempt $n FAILED $(date +%T)"
  done
  FAILED+=("$label"); return 1
}
main() { "$PY" -u 05_distill/main.py --teacher "$DISTILL_TEACHER" "$@"; }

STOP=1.3   # LOOP.md's --wsd-stop (README, "Step budget"); this chain ran 2026-10-05 with 1.85, the old rung-A spread
echo "[fillchain] --wsd-stop $STOP"

attempt B-fast32h16-56k main --rung B --wsd-max 56000 --runs "fast32h16:a0.50:$BRH"
attempt C-yamnet-a0.25 main --rung C --wsd-max 56000 --wsd-stop "$STOP" --runs "yamnet:a0.25:select:$BRH"
attempt C-mels-fast32-twofast32 main --rung C --until cache_fe --runs "fast32:a0.50 twofast32:a0.50"
attempt C-fast32-a0.50 main --rung C --wsd-max 56000 --wsd-stop "$STOP" --runs "fast32:a0.50:$BRH"
attempt C-twofast32-a0.50 main --rung C --wsd-max 56000 --wsd-stop "$STOP" --runs "twofast32:a0.50:$BRH"
attempt C-fast32h16-s2 main --rung C --seed 2 --wsd-max 28000 --wsd-halvings 2 --runs "fast32h16:a0.50:$BRH"

(cd 05_distill && "$PY" ladder_record.py proxy) || true
for t in fe_B_fast32h16_a0.50_s1_c-buzz-rain-human_wsd fe_C_fast32h16_a0.50_s1_c-buzz-rain-human_wsd \
         fe_C_fast32h16_a0.50_s2_c-buzz-rain-human_wsd fe_C_yamnet_a0.25_s1_select_c-buzz-rain-human_wsd \
         fe_C_fast32_a0.50_s1_c-buzz-rain-human_wsd fe_C_twofast32_a0.50_s1_c-buzz-rain-human_wsd; do
  (cd 05_distill && "$PY" ladder_record.py wsd --name "$t") || true
done
"$PY" tools/human/frontier_svg.py || true
"$PY" tools/human/frontier_html.py || true
"$PY" tools/human/wsd_svg.py --tol "$STOP" || true
echo "[fillchain] failed steps: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
