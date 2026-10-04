#!/bin/bash
# WSD follow-ups to the rung-C curve (2026-10-03, HANDOFF.md). Steps run in order on the one GPU; a failed step
# is retried once (main.py resumes), then the chain moves on to the next. Re-running the chain resumes it.
#   1 equivalence: rung B wsd7000 vs the cosine 7k runs, yamnet + fast32h16 a0.50
#   2 fast32h16 a0.50 rung C to 112k (headline still rising at 56k; no --wsd-stop, hit% dipped at 56k)
#   3 yamnet a0.50 rung C to 28k (the stop at 14k fired on a -1.38 drop, inside the 1.31 noise)
#   4 rung-B curve on the frontier student yamnet a0.25
#   5 fast32h16 a0.25 rung C to 56k (does the faster, smaller student also keep rising?)
#
#   tools/launch_job.sh 05_distill/data/chain_wsd.log -- bash 05_distill/chain_wsd.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source tools/python_path.sh   # PY
export PYTHONUNBUFFERED=1
BRH=classes=ins_buzz+ambient_rain+human
FAILED=()

step() {  # step <label> <main.py args...>
  local label=$1; shift
  for attempt in 1 2; do
    echo "[wsdchain] $label: attempt $attempt start $(date +%T)"
    if "$PY" -u 05_distill/main.py --teacher v4-ft-ps-e60-moderate "$@"; then
      echo "[wsdchain] $label: done $(date +%T)"; return 0
    fi
    echo "[wsdchain] $label: attempt $attempt FAILED $(date +%T)"
  done
  FAILED+=("$label")
}

step equivalence --rung B --wsd 7000 --runs "yamnet:a0.50:select:$BRH fast32h16:a0.50:$BRH"
step fast32h16-C-112k --rung C --wsd-max 112000 --wsd-halvings 4 --runs "fast32h16:a0.50:$BRH"
step yamnet-C-28k --rung C --wsd 7000,14000,28000 --runs "yamnet:a0.50:select:$BRH"
step yamnet-a0.25-B-curve --rung B --wsd 7000,14000,28000,70000 --runs "yamnet:a0.25:select:$BRH"
step fast32h16-a0.25-C-56k --rung C --wsd-max 56000 --runs "fast32h16:a0.25:$BRH"

echo "[wsdchain] failed steps: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
