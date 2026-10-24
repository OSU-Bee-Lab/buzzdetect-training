#!/usr/bin/env bash
# batch 43: ps2sf-d8-r2 (repeat draw, existing cache), pssf-d8 and ps12sf-d8 (new embedders).
# Every step resumes on rerun (stage 2 skips extracted idents, stage 3 skips finished folds).
set -uo pipefail
WT=/home/luke/projects/buzzdetect-training/.local/worktrees
PY=$(bash /home/luke/projects/buzzdetect-training/tools/python_path.sh)
export MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1 BUZZDETECT_CHUNK_FRAMES=48
SMOKE_FOLD="Luke - Diel Drivers/2026-04-08/1_150"
D8=(TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024)
D8_3V=(TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=512 TRUNK_ACCUM=2 SCORE_CHUNK=512)

extract() {  # extract <slug> <emb> <set>
  cd "$WT/$1" || return 1
  echo "[chain] $1: $3 extraction of $2"
  "$PY" -u 02_set/main.py --set $3 --embedder $2 --workers 1 > extract_$3.log 2>&1 \
    || { echo "[chain] $1: $3 extraction FAILED, see $WT/$1/extract_$3.log"; tail -20 extract_$3.log; return 1; }
  tail -1 extract_$3.log
}

smoke() {  # smoke <slug> <emb> <env...>
  local slug=$1 emb=$2; shift 2
  cd "$WT/$slug" || return 1
  [ -f "models/$slug/folds_sx.csv" ] && return 0
  [ -f ".smoke_ok" ] && return 0
  echo "[chain] $slug: medium smoke"
  rm -rf "models/test_${slug}_smoke"
  env "$@" "$PY" -u 03_train/main.py --name "test_${slug}_smoke" --set medium --embedder $emb \
      --translation general --epochs 1 --only-folds "$SMOKE_FOLD" -y --no-surprisal > smoke.log 2>&1 \
    || { echo "[chain] $slug: SMOKE FAILED, see $WT/$slug/smoke.log"; tail -20 smoke.log; return 1; }
  grep -E "buzz logit SD" smoke.log | head -2
  rm -rf "models/test_${slug}_smoke"
  touch .smoke_ok
}

cv() {  # cv <slug> <emb> <env...>
  local slug=$1 emb=$2; shift 2
  cd "$WT/$slug" || return 1
  echo "[chain] $slug: medium CV"
  env "$@" "$PY" -u 03_train/main.py --name "$slug" --set medium --embedder $emb \
      --translation general --epochs 30 -y --verbose 2>&1 | tee -a train.log | grep --line-buffered -vE "━|step - "
  local rc=${PIPESTATUS[0]}
  echo "[chain] $slug: exit $rc"
  return $rc
}

SF2=yamnet_trunk_pitchshift2sf_depth8
SF1=yamnet_trunk_pitchshiftsf_depth8
SF12=yamnet_trunk_pitchshift12sf_depth8
extract pssf-d8 $SF1 lite || exit 1
extract ps12sf-d8 $SF12 lite || exit 1
extract pssf-d8 $SF1 medium || exit 1
extract ps12sf-d8 $SF12 medium || exit 1
smoke pssf-d8 $SF1 "${D8[@]}" || exit 1
smoke ps12sf-d8 $SF12 "${D8_3V[@]}" || exit 1
cv ps2sf-d8-r2 $SF2 "${D8_3V[@]}" || exit 1
cv pssf-d8 $SF1 "${D8[@]}" || exit 1
cv ps12sf-d8 $SF12 "${D8_3V[@]}" || exit 1
echo "[chain] all done"
