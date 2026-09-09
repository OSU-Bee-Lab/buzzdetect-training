#!/usr/bin/env bash
# Probe-convergence grid (IDEAS.md L1-L7), one lever at a time off cv_baseline.
# Every run is `03_train/main.py --set medium --embedder yamnet --translation
# general` plus exactly one non-default probe flag.
PY=/home/luke/anaconda3/envs/buzzdetect-train/bin/python
cd "$(dirname "$0")"

run () {
  name="$1"; shift
  if [ -f "models/$name/folds_sx.csv" ]; then echo "== $name already done"; return; fi
  echo "== $(date +%H:%M:%S) $name : $*"
  env PYTHONUNBUFFERED=1 BUZZDETECT_NO_GPU=1 "$PY" -u 03_train/main.py \
      --name "$name" --set medium --embedder yamnet --translation general \
      --no-surprisal -y "$@" > "grid_$name.log" 2>&1
  echo "== $(date +%H:%M:%S) $name done rc=$?"
  tail -2 "models/$name/folds_sx.csv" 2>/dev/null
}

run L1_sens        --monitor val_sens
run L2_mindelta0   --min-delta 0
run L3_batch4096   --batch-size 4096
run L5_drop0       --dropout 0
run L5_drop01      --dropout 0.1
run L6_ls0         --label-smoothing 0
run L6_ls005       --label-smoothing 0.05
run L7_wd1e4       --weight-decay 0.0001
echo "== GRID COMPLETE $(date)"
