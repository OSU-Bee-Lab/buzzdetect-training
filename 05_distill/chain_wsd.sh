#!/bin/bash
# WSD follow-ups to the rung-C curve, with the stop rule on hit@K (2026-10-03, HANDOFF.md). Steps run in order on
# the one GPU; a failed step is retried once (main.py resumes), then the chain moves on. Re-running resumes it.
#   0 backfill hit@K on every finished run; print the readout-vs-headline proxy check and the two rung-C curves
#   1 equivalence: rung B wsd7000 vs the cosine 7k runs, yamnet + fast32h16 a0.50
#   2 yamnet a0.50 rung C to 56k under the new rule (the old rule stopped it at 14k on a -1.38 drop in hit% at 0)
#   3 fast32h16 a0.50 rung C to 112k, no stop: a check on the new rule (hit% at 0 dipped at 56k, the headline rose)
#   4 rung-B curve on the frontier student yamnet a0.25, new rule
#   5 fast32h16 a0.25 rung C to 56k, new rule (does the faster, smaller student also keep rising?)
# The --wsd-stop tolerance is the rung-A repeat spread of hit@K (ladder_record.py spread), measured in step 0.
#
#   tools/launch_job.sh 05_distill/data/chain_wsd.log -- bash 05_distill/chain_wsd.sh
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
    echo "[wsdchain] $label: attempt $n start $(date +%T)"
    if "$@"; then echo "[wsdchain] $label: done $(date +%T)"; return 0; fi
    echo "[wsdchain] $label: attempt $n FAILED $(date +%T)"
  done
  FAILED+=("$label"); return 1
}
step() { local label=$1; shift; attempt "$label" "$PY" -u 05_distill/main.py --teacher "$DISTILL_TEACHER" "$@"; }
curve() { "$PY" 05_distill/ladder_record.py wsd --name "$1" || true; }

attempt backfill-hitk "$PY" -u 05_distill/backfill_hitk.py
"$PY" 05_distill/ladder_record.py proxy || true
curve fe_C_yamnet_a0.50_s1_select_c-buzz-rain-human_wsd
curve fe_C_fast32h16_a0.50_s1_c-buzz-rain-human_wsd
STOP=$("$PY" 05_distill/ladder_record.py spread) || { echo "[wsdchain] no hit@K spread; stopping"; exit 1; }
echo "[wsdchain] --wsd-stop $STOP (rung-A hit@K spread)"

step equivalence --rung B --wsd 7000 --runs "yamnet:a0.50:select:$BRH fast32h16:a0.50:$BRH"
step yamnet-C-56k --rung C --wsd-max 56000 --wsd-stop "$STOP" --runs "yamnet:a0.50:select:$BRH"
step fast32h16-C-112k --rung C --wsd-max 112000 --wsd-halvings 4 --runs "fast32h16:a0.50:$BRH"
step yamnet-a0.25-B-curve --rung B --wsd 7000,14000,28000,70000 --wsd-stop "$STOP" --runs "yamnet:a0.25:select:$BRH"
step fast32h16-a0.25-C-56k --rung C --wsd-max 56000 --wsd-stop "$STOP" --runs "fast32h16:a0.25:$BRH"

"$PY" 05_distill/ladder_record.py proxy || true
echo "[wsdchain] failed steps: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
