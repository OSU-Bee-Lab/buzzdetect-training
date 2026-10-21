#!/usr/bin/env bash
# batch 41: binary-d8, plain-d8, sam-d8 (ps-depth8 cache), then ps2-seamfree-d8 (new embedder).
# `chain_b41.sh smoke` runs only the 1-epoch smokes; no argument runs smokes then CVs.
# Every step resumes on rerun (stage 2 skips extracted idents, stage 3 skips finished folds).
set -uo pipefail
MODE=${1:-full}
WT=/home/luke/projects/buzzdetect-training/.local/worktrees
PY=$(bash /home/luke/projects/buzzdetect-training/tools/python_path.sh)
export MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1 BUZZDETECT_CHUNK_FRAMES=48
SMOKE_FOLD="Luke - Diel Drivers/2026-04-08/1_150"
D8=(TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024)
D8_3V=(TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=512 TRUNK_ACCUM=2 SCORE_CHUNK=512)

smoke() {  # smoke <slug> <emb> <translation> <env...>
  local slug=$1 emb=$2 tr=$3; shift 3
  cd "$WT/$slug" || return 1
  [ -f "models/$slug/folds_sx.csv" ] && return 0
  [ -f ".smoke_ok" ] && return 0
  echo "[chain] $slug: medium smoke"
  rm -rf "models/test_${slug}_smoke"
  env "$@" "$PY" -u 03_train/main.py --name "test_${slug}_smoke" --set medium --embedder $emb \
      --translation $tr --epochs 1 --only-folds "$SMOKE_FOLD" -y --no-surprisal > smoke.log 2>&1 \
    || { echo "[chain] $slug: SMOKE FAILED, see $WT/$slug/smoke.log"; tail -20 smoke.log; return 1; }
  grep -E "\[pin\]|SAM|view 0|buzz logit SD" smoke.log | head -4
  rm -rf "models/test_${slug}_smoke"
  touch .smoke_ok
}

cv() {  # cv <slug> <emb> <translation> <env...>
  local slug=$1 emb=$2 tr=$3; shift 3
  cd "$WT/$slug" || return 1
  echo "[chain] $slug: medium CV"
  env "$@" "$PY" -u 03_train/main.py --name "$slug" --set medium --embedder $emb \
      --translation $tr --epochs 30 -y --verbose 2>&1 | tee train.log | grep --line-buffered -vE "━|step - "
  local rc=${PIPESTATUS[0]}
  echo "[chain] $slug: exit $rc"
  return $rc
}

PS8=yamnet_trunk_pitchshift_depth8
SF=yamnet_trunk_pitchshift2sf_depth8
smoke binary-d8 $PS8 binary "${D8[@]}" PIN_BUZZ_TRANSLATION=general || exit 1
smoke plain-d8 $PS8 general "${D8[@]}" TRUNK_ONLY_VIEW=0 || exit 1
smoke sam-d8 $PS8 general "${D8[@]}" TRUNK_SAM_RHO=0.05 || exit 1
if [ "$MODE" = smoke ]; then
  cd "$WT/ps2-seamfree-d8" && echo "[chain] ps2-seamfree-d8: lite extraction smoke"
  "$PY" -u 02_set/main.py --set lite --embedder $SF --workers 1 > smoke_extract.log 2>&1 \
    || { echo "[chain] lite extraction FAILED"; tail -20 smoke_extract.log; exit 1; }
  tail -2 smoke_extract.log
  echo "[chain] smokes done"; exit 0
fi

cd "$WT/ps2-seamfree-d8" && echo "[chain] ps2-seamfree-d8: medium extraction"
"$PY" -u 02_set/main.py --set medium --embedder $SF --workers 1 > extract.log 2>&1 \
  || { echo "[chain] medium extraction FAILED"; tail -20 extract.log; exit 1; }
tail -1 extract.log
cv binary-d8 $PS8 binary "${D8[@]}" PIN_BUZZ_TRANSLATION=general || exit 1
cv plain-d8 $PS8 general "${D8[@]}" TRUNK_ONLY_VIEW=0 || exit 1
cv sam-d8 $PS8 general "${D8[@]}" TRUNK_SAM_RHO=0.05 || exit 1
smoke ps2-seamfree-d8 $SF general "${D8_3V[@]}" || exit 1
cv ps2-seamfree-d8 $SF general "${D8_3V[@]}" || exit 1
echo "[chain] all done"
