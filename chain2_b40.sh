#!/usr/bin/env bash
# batch 40 chain 2: gain-aug-d8 then pitch-aug-d8, after chain 1 on medium, depth8 recipe. Each step resumes on rerun
# (augment.py skips cached files, stage 3 skips finished folds); a failed smoke stops the chain.
set -uo pipefail
WT=/home/luke/projects/buzzdetect-training/.local/worktrees
PY=$(bash /home/luke/projects/buzzdetect-training/tools/python_path.sh)
export MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1 BUZZDETECT_CHUNK_FRAMES=48
EMB=yamnet_trunk_pitchshift_depth8
SMOKE_FOLD="Luke - Diel Drivers/2026-04-08/1_150"
D8=(TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024)

run() {  # run <slug> <aug_dirname> -- <augment.py spec args>
  local slug=$1 aug=$2; shift 3
  cd "$WT/$slug" || return 1
  if [ ! -f "models/$slug/folds_sx.csv" ]; then
    echo "[chain] $slug: augment ($*)"
    "$PY" -u 02_set/augment.py --set medium --embedder $EMB --all-folds "$@" > augment.log 2>&1 \
      || { echo "[chain] $slug: AUGMENT FAILED"; tail -20 augment.log; exit 1; }
    tail -1 augment.log
    echo "[chain] $slug: medium smoke"
    rm -rf "models/test_${slug}_smoke"
    env "${D8[@]}" "$PY" -u 03_train/main.py --name "test_${slug}_smoke" --set medium --embedder $EMB \
        --augment "$aug" --translation general --epochs 1 --only-folds "$SMOKE_FOLD" -y --no-surprisal > smoke.log 2>&1 \
      || { echo "[chain] $slug: SMOKE FAILED, see $WT/$slug/smoke.log"; tail -20 smoke.log; exit 1; }
    grep -E "training on|sens@fpr" smoke.log | head -3
    rm -rf "models/test_${slug}_smoke"
  fi
  echo "[chain] $slug: medium CV"
  env "${D8[@]}" "$PY" -u 03_train/main.py --name "$slug" --set medium --embedder $EMB --augment "$aug" \
      --translation general --epochs 30 -y --verbose 2>&1 | tee train.log | grep --line-buffered -vE "━|step - "
  local rc=${PIPESTATUS[0]}
  echo "[chain] $slug: exit $rc"
  return $rc
}

# wait for chain 1 (eq-aug-d8, snr-aug-d8; launch_job pid 726415) to free the GPU
while kill -0 726415 2>/dev/null; do sleep 60; done
echo "[chain] chain 1 ended"
run gain-aug-d8 augment_gain10.0_p0.5_s0 -- --gain 10 || exit 1
run pitch-aug-d8 augment_pitch2.0_p0.5_s0 -- --pitch 2 || exit 1
echo "[chain] all done"
