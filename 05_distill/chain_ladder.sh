#!/bin/bash
# The distillation ladder: A/seed1, A/seed2, B/seed1, each train -> V flips ->
# export -> eval on the 5 rotating folds -> ladder.jsonl line; then the table and
# the speed comparison for the rung-B model. Stops on the first failure; every
# stage skips if its artifact exists, so relaunching resumes.
#
#   tools/launch_job.sh <log> -- bash 05_distill/chain_ladder.sh
#
# STEPS is the one budget for every rung (05_distill/LADDER.md).
set -euo pipefail
STEPS=${STEPS:-12000}
WT=/home/luke/projects/buzzdetect-training/.claude/worktrees/distill-lite
MAIN=/home/luke/projects/buzzdetect-training
L=$MAIN/.local/distill
ENGINE_PY=/home/luke/projects/buzzdetect/engine/.venv/bin/python3
source "$WT/tools/python_path.sh"   # sets PY
cd "$WT"
export PYTHONUNBUFFERED=1
mkdir -p "$L/runs" "$L/eval" "$L/models"

run_one() {
  local rung=$1 seed=$2
  local name=lad_${rung}_s${seed}
  local wallf=$L/runs/$name/wall.txt
  mkdir -p "$L/runs/$name"
  [ -f "$wallf" ] || echo 0 > "$wallf"
  stage() {  # stage <label> <artifact> <cmd...>: run cmd unless artifact exists; add wall time
    local label=$1 art=$2; shift 2
    if [ -e "$art" ]; then echo "[chain] $name $label: done (skip)"; return; fi
    echo "[chain] $name $label: start $(date +%T)"
    local t0=$(date +%s)
    "$@"
    echo $(( $(cat "$wallf") + $(date +%s) - t0 )) > "$wallf"
    echo "[chain] $name $label: done $(date +%T)"
  }
  stage train "$L/runs/$name/TRAIN_DONE" bash -c \
    "'$PY' -u 05_distill/distill_train.py --rung $rung --steps $STEPS --batch 512 --seed $seed --name $name --eval-every 2000 && touch '$L/runs/$name/TRAIN_DONE'"
  stage export "$L/models/$name/model.onnx" "$PY" -u 05_distill/export_student.py export --run "$name" --name "$name"
  stage eval "$L/eval/$name/folds_sx.csv" "$PY" -u 05_distill/eval_folds.py run --check-labels \
    --onnx "$L/models/$name/model.onnx" --out "$L/eval/$name"
  if ! grep -q "\"name\": \"$name\"" "$L/ladder.jsonl" 2>/dev/null; then
    "$PY" 05_distill/ladder_record.py record --rung "$rung" --seed "$seed" --steps "$STEPS" \
      --name "$name" --wall "$(cat "$wallf")"
  fi
}

run_one A 1
run_one A 2
run_one B 1
"$PY" 05_distill/ladder_record.py table
echo "[chain] speed: rung-B model, 15 repeats, 20 s and 200 s, vs yamnet_large_general"
"$ENGINE_PY" 05_distill/export_student.py time --name lad_B_s1 --repeats 15
echo "[chain] all done"
