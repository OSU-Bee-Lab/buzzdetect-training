#!/usr/bin/env bash
# batch 42, part 2: waits for chain_b42 (pid in $1), then bntrain-d8 CV (trainable tail BN,
# saved fold weights) and the offline site-adaptive / shrunk BN rescoring (bn-site-d8, bn-shrunk-d8).
# Resumes on rerun: stage 3 skips finished folds, soup.py skips scored folds.
set -uo pipefail
[ -n "${1:-}" ] && while kill -0 "$1" 2>/dev/null; do sleep 60; done
cd /home/luke/projects/buzzdetect-training/.local/worktrees/bn-site-d8 || exit 1
PY=$(bash /home/luke/projects/buzzdetect-training/tools/python_path.sh)
export MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1 BUZZDETECT_CHUNK_FRAMES=48
# batch 1024 with trainable BN OOMs the 4 GB card (smoke, 2026-10-23): 512 x 2 accumulated keeps the update size; BN batch stats per 512
export TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=512 TRUNK_ACCUM=2 TRUNK_BN_TRAIN=1 TRUNK_SAVE_WEIGHTS=1
SMOKE_FOLD="Luke - Diel Drivers/2026-04-08/1_150"
EMB=yamnet_trunk_pitchshift_depth8
if [ ! -f .smoke_ok ] && [ ! -f models/bntrain-d8/folds_sx.csv ]; then
  echo "[chain] bntrain-d8: medium smoke + soup --bn-site smoke"
  rm -rf models/test_bntrain_smoke models/test_bnsite_smoke
  "$PY" -u 03_train/main.py --name test_bntrain_smoke --set medium --embedder $EMB --translation general \
      --epochs 1 --only-folds "$SMOKE_FOLD" -y --no-surprisal > smoke.log 2>&1 \
    && "$PY" -u tools_exp/soup.py test_bnsite_smoke --members models/test_bntrain_smoke --bn-site 20000 --bn-shrink 6000 >> smoke.log 2>&1 \
    || { echo "[chain] SMOKE FAILED, see smoke.log"; tail -20 smoke.log; exit 1; }
  grep -E "buzz logit SD|BN layers re-estimated" smoke.log | head -3
  rm -rf models/test_bntrain_smoke models/test_bnsite_smoke
  touch .smoke_ok
fi
echo "[chain] bntrain-d8: medium CV"
"$PY" -u 03_train/main.py --name bntrain-d8 --set medium --embedder $EMB --translation general \
    --epochs 30 -y --verbose 2>&1 | tee -a train.log | grep --line-buffered -vE "━|step - "
rc=${PIPESTATUS[0]}; echo "[chain] bntrain-d8: exit $rc"; [ $rc = 0 ] || exit $rc
echo "[chain] bn-site-d8: offline rescoring"
"$PY" -u tools_exp/soup.py bn-site-d8 --members models/bntrain-d8 --bn-site 20000 2>&1 | tee -a soup.log | grep --line-buffered -v "mean |shift|" || exit 1
echo "[chain] bn-shrunk-d8: offline rescoring"
"$PY" -u tools_exp/soup.py bn-shrunk-d8 --members models/bntrain-d8 --bn-site 20000 --bn-shrink 6000 2>&1 | tee -a soup.log | grep --line-buffered -v "mean |shift|" || exit 1
echo "[chain] all done"
