#!/usr/bin/env bash
# batch 39: smoke then run mixstyle-xfold, pauc-l01 on medium in turn.
# Each step resumes on rerun (stage 3 skips finished folds); a failed smoke stops the chain.
set -uo pipefail
WT=/home/luke/projects/buzzdetect-training/.local/worktrees
PY=$(bash /home/luke/projects/buzzdetect-training/tools/python_path.sh)
export MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1
SMOKE_FOLD="Luke - Diel Drivers/2026-04-08/1_150"

run() {  # run <slug> <name> <env...> -- <extra args>
  local slug=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  cd "$WT/$slug" || return 1
  if [ ! -f "models/$slug/folds_sx.csv" ]; then
    echo "[chain] $slug: lite smoke"
    rm -rf "models/test_${slug}_smoke"
    env "${envs[@]}" "$PY" -u 03_train/main.py --name "test_${slug}_smoke" --set lite "$@" \
        --translation general --epochs 1 --only-folds "$SMOKE_FOLD" -y --no-surprisal > smoke.log 2>&1 \
      || { echo "[chain] $slug: SMOKE FAILED, see $WT/$slug/smoke.log"; tail -20 smoke.log; exit 1; }
    grep -E "MixStyle|pAUC|PAUC|sens@fpr" smoke.log | head -5
    rm -rf "models/test_${slug}_smoke"
  fi
  echo "[chain] $slug: medium CV"
  env "${envs[@]}" "$PY" -u 03_train/main.py --name "$slug" --set medium "$@" \
      --translation general --epochs 30 -y --verbose 2>&1 | tee train.log | grep --line-buffered -vE "━|step - "
  local rc=${PIPESTATUS[0]}
  echo "[chain] $slug: exit $rc"
  return $rc
}

D8=(TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024)
run mixstyle-xfold "${D8[@]}" TRUNK_MIXSTYLE=0.5 TRUNK_MIXSTYLE_XFOLD=1 -- --embedder yamnet_trunk_pitchshift_depth8 || exit 1
run pauc-l01 "${D8[@]}" TRAIN_PAUC_LAMBDA=0.1 -- --embedder yamnet_trunk_pitchshift_depth8 || exit 1
echo "[chain] all done"
