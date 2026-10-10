#!/usr/bin/env bash
# Runs site-adv-w03 then site-adv-w10 after site-adv-w01's job (pid $1) exits.
# Each stage-3 call resumes on rerun (finished folds are skipped).
WT=/home/luke/projects/buzzdetect-training/.local/worktrees
while kill -0 "$1" 2>/dev/null; do sleep 30; done
for pair in "w03:0.3" "w10:1.0"; do
  w=${pair%%:*}; lam=${pair##*:}
  cd "$WT/site-adv-$w" || exit 1
  TRUNK_ADV=$lam TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 MALLOC_ARENA_MAX=2 \
    python -u 03_train/main.py --name site-adv-$w --set medium \
    --embedder yamnet_trunk_pitchshift_depth8 --translation general --epochs 30 -y --verbose
  echo "[chain] site-adv-$w exit $?"
done
