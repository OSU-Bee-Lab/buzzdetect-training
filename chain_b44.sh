#!/usr/bin/env bash
# batch 44: quietup3-d8, quietup6-d8, focal-d8, mixup-d8 -- code changes on the
# ps-depth8 recipe, existing yamnet_trunk_pitchshift_depth8 cache, no extraction.
# Every step resumes on rerun (smoke markers, stage 3 skips finished folds).
set -uo pipefail
WT=/home/luke/projects/buzzdetect-training/.local/worktrees
PY=$(bash /home/luke/projects/buzzdetect-training/tools/python_path.sh)
export MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1 BUZZDETECT_CHUNK_FRAMES=48
SMOKE_FOLD="Luke - Diel Drivers/2026-04-08/1_150"
EMB=yamnet_trunk_pitchshift_depth8
D8=(TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024)

smoke() {  # smoke <slug>
  local slug=$1
  cd "$WT/$slug" || return 1
  [ -f "models/$slug/folds_sx.csv" ] && return 0
  [ -f ".smoke_ok" ] && return 0
  echo "[chain] $slug: medium smoke"
  rm -rf "models/test_${slug}_smoke"
  env "${D8[@]}" "$PY" -u 03_train/main.py --name "test_${slug}_smoke" --set medium --embedder $EMB \
      --translation general --epochs 1 --only-folds "$SMOKE_FOLD" -y --no-surprisal > smoke.log 2>&1 \
    || { echo "[chain] $slug: SMOKE FAILED, see $WT/$slug/smoke.log"; tail -20 smoke.log; return 1; }
  grep -E "buzz logit SD|\[quietup\]" smoke.log | head -3
  rm -rf "models/test_${slug}_smoke"
  touch .smoke_ok
}

cv() {  # cv <slug>
  local slug=$1
  cd "$WT/$slug" || return 1
  echo "[chain] $slug: medium CV"
  env "${D8[@]}" "$PY" -u 03_train/main.py --name "$slug" --set medium --embedder $EMB \
      --translation general --epochs 30 -y --verbose 2>&1 | tee -a train.log | grep --line-buffered -vE "━|step - "
  local rc=${PIPESTATUS[0]}
  echo "[chain] $slug: exit $rc"
  return $rc
}

for s in quietup3-d8 quietup6-d8 focal-d8 mixup-d8; do smoke $s || exit 1; done
for s in quietup3-d8 quietup6-d8 focal-d8 mixup-d8; do cv $s || exit 1; done
echo "[chain] all done"
