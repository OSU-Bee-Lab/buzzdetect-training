#!/usr/bin/env bash
# Targeted repeats, not a noise-floor study.
#
# One identical-config pair (L1_sens vs L1_sens_r2) put the run-to-run SD on the
# CV mean at ~0.016, so a single-run-vs-single-run difference has SE ~0.023 and
# can only resolve effects >= ~0.045. Properly resolving a +0.015 effect would
# take ~9 runs per arm. These three are the cheap, targeted subset.
PY=/home/luke/anaconda3/envs/buzzdetect-train/bin/python
cd "$(dirname "$0")"

run () {
  name="$1"; shift
  if [ -f "models/$name/folds_sx.csv" ]; then echo "== $name already done"; return; fi
  echo "== $(date +%H:%M:%S) $name : ${*:-(defaults = cv_baseline config)}"
  env PYTHONUNBUFFERED=1 BUZZDETECT_NO_GPU=1 "$PY" -u 03_train/main.py \
      --name "$name" --set medium --embedder yamnet --translation general \
      --no-surprisal -y "$@" > "grid_$name.log" 2>&1
  echo "== $(date +%H:%M:%S) $name done rc=$?"
  tail -1 "models/$name/folds_sx.csv" 2>/dev/null
}

# Baseline repeats. cv_baseline is the denominator of every comparison in this
# era; if it is an unlucky draw, every delta in log.jsonl carries the same bias
# in the same direction. These give it n=3 so later work can compare to a mean.
# No flags => ProbeConfig defaults => cv_baseline's exact config.
run base_r2
run base_r3

# The one live claim sitting at exactly one run-to-run spread (+0.015).
run L6_ls005_r2 --label-smoothing 0.05

echo "== NOISE COMPLETE $(date)"
