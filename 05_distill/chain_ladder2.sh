#!/bin/bash
# Ladder chain 2 (same rules as chain_ladder.sh: stop on failure, resumable, one
# ladder.jsonl line per run, 7000 steps, eval after every run):
#   1 cache rung C          2 train C (a0.50, seed 1, in-memory loader)
#   3 variants at rung B, seed 1: a0.50_d12, a0.375 (each timed 20 s / 200 s right after its eval)
#   3b streaming-loader sanity: rung B through shards, must match in-memory B within the A spread
#   gate: rung D only if C beat B by more than the A spread (FORCE_D=1 overrides)
#   4 cache rung D, pack shards A..D     5 train D (a0.50, stream), and D with the qualifying variant, if any
#
#   tools/launch_job.sh <log> -- bash 05_distill/chain_ladder2.sh
set -euo pipefail
STEPS=${STEPS:-7000}
WT=/home/luke/projects/buzzdetect-training/.claude/worktrees/distill-lite
MAIN=/home/luke/projects/buzzdetect-training
L=$MAIN/.local/distill
ENGINE_PY=/home/luke/projects/buzzdetect/engine/.venv/bin/python3
ONNX_PY=$MAIN/.local/venv-onnx/bin/python
source "$WT/tools/python_path.sh"   # PY
cd "$WT"
export PYTHONUNBUFFERED=1

stage_cmd() {  # stage_cmd <wallfile> <label> <artifact> <cmd...>
  local wallf=$1 label=$2 art=$3; shift 3
  if [ -e "$art" ]; then echo "[chain] $label: done (skip)"; return; fi
  echo "[chain] $label: start $(date +%T)"
  local t0=$(date +%s)
  "$@"
  echo $(( $(cat "$wallf") + $(date +%s) - t0 )) > "$wallf"
  echo "[chain] $label: done $(date +%T)"
}

# run_one <rung> <seed> <arch> <loader>
run_one() {
  local rung=$1 seed=$2 arch=$3 loader=$4
  local name=lad_${rung}
  [ "$arch" = a0.50 ] || name=${name}_${arch}
  name=${name}_s${seed}
  [ "$loader" = mem ] || name=${name}_${loader}
  local d=$L/runs/$name wallf=$L/runs/$name/wall.txt
  mkdir -p "$d" "$L/eval" "$L/models"
  [ -f "$wallf" ] || echo 0 > "$wallf"
  stage_cmd "$wallf" "$name train" "$d/TRAIN_DONE" bash -c \
    "'$PY' -u 05_distill/distill_train.py --rung $rung --steps $STEPS --batch 512 --seed $seed --name $name --arch $arch --loader $loader --eval-every 2000 && touch '$d/TRAIN_DONE'"
  stage_cmd "$wallf" "$name export" "$L/models/$name/model.onnx" "$PY" -u 05_distill/export_student.py export --run "$name" --name "$name"
  stage_cmd "$wallf" "$name eval" "$L/eval/$name/folds_sx.csv" "$PY" -u 05_distill/eval_folds.py run --check-labels \
    --onnx "$L/models/$name/model.onnx" --out "$L/eval/$name"
  if [ "$arch" != a0.50 ] || [ "$rung" = D ]; then
    stage_cmd "$wallf" "$name speed" "$L/models/$name/speed_200.json" "$ENGINE_PY" 05_distill/export_student.py time --name "$name" --repeats 15
  fi
  if ! grep -q "\"name\": \"$name\"" "$L/ladder.jsonl" 2>/dev/null; then
    "$PY" 05_distill/ladder_record.py record --rung "$rung" --seed "$seed" --steps "$STEPS" --name "$name" \
      --wall "$(cat "$wallf")" --arch "$arch" --loader "$loader"
  fi
}

# 1. cache C
if [ ! -e "$L/CACHE_C_DONE" ]; then
  echo "[chain] cache C: start $(date +%T)"; "$ONNX_PY" 05_distill/cache.py --rung C; touch "$L/CACHE_C_DONE"
fi
# 2. train C
run_one C 1 a0.50 mem
# 3. variants at rung B
run_one B 1 a0.50_d12 mem
run_one B 1 a0.375 mem
# 3b. streaming sanity
"$PY" -u 05_distill/shards.py pack --rung B
run_one B 1 a0.50 stream
"$PY" 05_distill/ladder_record.py streamcheck
# gate on the stopping rule
if ! "$PY" 05_distill/ladder_record.py gate; then
  if [ "${FORCE_D:-0}" != 1 ]; then "$PY" 05_distill/ladder_record.py table; echo "[chain] stopped by the ladder rule (FORCE_D=1 to override)"; exit 0; fi
fi
# 4. cache D + shards
if [ ! -e "$L/CACHE_D_DONE" ]; then
  echo "[chain] cache D: start $(date +%T)"; "$ONNX_PY" 05_distill/cache.py --rung D; touch "$L/CACHE_D_DONE"
fi
"$PY" -u 05_distill/shards.py pack --rung D
# 5. train D
run_one D 1 a0.50 stream
V=$("$PY" 05_distill/ladder_record.py decide)
echo "[chain] qualifying variant for D: $V"
if [ "$V" != none ]; then run_one D 1 "$V" stream; fi
"$PY" 05_distill/ladder_record.py table
echo "[chain] all done"
