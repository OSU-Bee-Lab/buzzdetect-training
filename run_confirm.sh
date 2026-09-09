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

run L1_sens_r2       --monitor val_sens
echo "== CONFIRM COMPLETE $(date)"
