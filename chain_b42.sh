#!/usr/bin/env bash
# batch 42: pitch-aux-d8, noquiet-d8 (both on the ps-depth8 cache, ps-depth8-r3's recipe).
# Every step resumes on rerun (smokes skip once .smoke_ok exists, stage 3 skips finished folds).
set -uo pipefail
WT=/home/luke/projects/buzzdetect-training/.local/worktrees
PY=$(bash /home/luke/projects/buzzdetect-training/tools/python_path.sh)
export MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1 BUZZDETECT_CHUNK_FRAMES=48
SMOKE_FOLD="Luke - Diel Drivers/2026-04-08/1_150"
D8=(TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024)

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
  grep -E "\[pin\]|\[noquiet\]|buzz logit SD" smoke.log | head -4
  rm -rf "models/test_${slug}_smoke"
  touch .smoke_ok
}

cv() {  # cv <slug> <emb> <translation> <env...>
  local slug=$1 emb=$2 tr=$3; shift 3
  cd "$WT/$slug" || return 1
  echo "[chain] $slug: medium CV"
  env "$@" "$PY" -u 03_train/main.py --name "$slug" --set medium --embedder $emb \
      --translation $tr --epochs 30 -y --verbose 2>&1 | tee -a train.log | grep --line-buffered -vE "━|step - "
  local rc=${PIPESTATUS[0]}
  echo "[chain] $slug: exit $rc"
  return $rc
}

PS8=yamnet_trunk_pitchshift_depth8
smoke pitch-aux-d8 $PS8 general_pitch "${D8[@]}" || exit 1
smoke noquiet-d8 $PS8 general "${D8[@]}" || exit 1
cv pitch-aux-d8 $PS8 general_pitch "${D8[@]}" || exit 1
cv noquiet-d8 $PS8 general "${D8[@]}" || exit 1
echo "[chain] all done"
