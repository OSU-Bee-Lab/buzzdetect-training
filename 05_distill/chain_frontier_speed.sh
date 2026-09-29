#!/bin/bash
# NOTE (2026-09-29): pre-generalization chain. Hard-codes the single-teacher layout under .local/distill and the old
# _fe cache; kept as the record of the runs it made. New work: 05_distill/main.py (README.md), same runs, any teacher.
# Speed frontier over front ends (random weights, speed only): every front end alone, and the
# a0.50 / a0.375 trunks on it. 20 s and 200 s, GPU + CPU, into .local/distill/arch_fe.
#   tools/launch_job.sh <log> -- bash 05_distill/chain_frontier_speed.sh
set -euo pipefail
WT=/home/luke/projects/buzzdetect-training/.claude/worktrees/distill-lite
OUT=${OUT:-/home/luke/projects/buzzdetect-training/.local/distill/arch_fe}
ENGINE_PY=/home/luke/projects/buzzdetect/engine/.venv/bin/python3
PY=/home/luke/anaconda3/envs/buzzdetect-train/bin/python
cd "$WT"
FES=${FES:-"yamnet lo64 lo32 lo16 two64 two32 two16 fast32 fast16"}
TRUNKS=${TRUNKS:-"a0.50 a0.375"}
C=""
for f in $FES; do C="$C fe@$f"; for t in $TRUNKS; do C="$C $t@$f"; done; done
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES="" $PY -u 05_distill/bench_arch.py export --out "$OUT" --candidates $C
for s in 20 200; do
  $ENGINE_PY 05_distill/bench_arch.py time --out "$OUT" --seconds $s --candidates $C | tee "$OUT/time_$s.txt"
  cp "$OUT/results.json" "$OUT/results_$s.json"
done
