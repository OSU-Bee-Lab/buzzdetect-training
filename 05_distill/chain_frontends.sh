#!/bin/bash
# Front-end frontier at rung B (seed 1, 7000 steps, same recipe as the ladder's B runs).
# Question: how fast can the student go, and what does each front end cost in sensitivity?
#   1 cache the front ends' inputs for rung B and V (cache_fe.py: decode once, all specs)
#   2 train / export / eval / time / record each run, in priority order; stop on failure, resumable
# Control first: the YAMNet front end with the same 'select' init the other front ends must use
# (they cannot use the layer-wise refit), so init effects and front-end effects are separable.
#   tools/launch_job.sh <log> -- bash 05_distill/chain_frontends.sh
#   RUNS="two32:a0.50 fast16:a0.25" tools/launch_job.sh <log> -- bash 05_distill/chain_frontends.sh   # custom list
set -euo pipefail
STEPS=${STEPS:-7000}
WT=/home/luke/projects/buzzdetect-training/.claude/worktrees/distill-lite
MAIN=/home/luke/projects/buzzdetect-training
L=$MAIN/.local/distill
ENGINE_PY=/home/luke/projects/buzzdetect/engine/.venv/bin/python3
ONNX_PY=$MAIN/.local/venv-onnx/bin/python
PY=/home/luke/anaconda3/envs/buzzdetect-train/bin/python
cd "$WT"
export PYTHONUNBUFFERED=1

# <frontend>:<arch>[:<init tag>]  ; init tag 'select' = the control
DEFAULT_RUNS="yamnet:a0.50:select fast32:a0.50 fast32h16:a0.50 fast32h32:a0.50 twofast32:a0.50 fast32h16:a0.375 fast32h16:a0.25 fast32h32:a0.25 two32:a0.50 lo32:a0.50 twofast32:a0.375"
RUNS=${RUNS:-$DEFAULT_RUNS}
FES=$(for r in $RUNS; do echo "${r%%:*}"; done | grep -v '^yamnet$' | sort -u | paste -sd, -)

stage_cmd() {  # stage_cmd <wallfile> <label> <artifact> <cmd...>
  local wallf=$1 label=$2 art=$3; shift 3
  if [ -e "$art" ]; then echo "[chain] $label: done (skip)"; return; fi
  echo "[chain] $label: start $(date +%T)"
  local t0=$(date +%s)
  "$@"
  echo $(( $(cat "$wallf") + $(date +%s) - t0 )) > "$wallf"
  echo "[chain] $label: done $(date +%T)"
}

run_fe() {  # run_fe <frontend> <arch> [<init>]
  local fe=$1 arch=$2 init=${3:-}
  local name=fe_B_${fe}_${arch}_s1
  [ -z "$init" ] || name=${name}_${init}
  local d=$L/runs/$name wallf=$L/runs/$name/wall.txt
  mkdir -p "$d" "$L/eval" "$L/models"
  [ -f "$wallf" ] || echo 0 > "$wallf"
  local initarg=""; [ -z "$init" ] || initarg="--init $init"
  stage_cmd "$wallf" "$name train" "$d/TRAIN_DONE" bash -c \
    "'$PY' -u 05_distill/distill_train.py --rung B --steps $STEPS --batch 512 --seed 1 --name $name --arch $arch --frontend $fe $initarg --eval-every 2000 && touch '$d/TRAIN_DONE'"
  stage_cmd "$wallf" "$name export" "$L/models/$name/model.onnx" "$PY" -u 05_distill/export_student.py export --run "$name" --name "$name"
  stage_cmd "$wallf" "$name eval" "$L/eval/$name/folds_sx.csv" "$PY" -u 05_distill/eval_folds.py run --check-labels \
    --onnx "$L/models/$name/model.onnx" --out "$L/eval/$name"
  stage_cmd "$wallf" "$name speed" "$L/models/$name/speed_200.json" "$ENGINE_PY" 05_distill/export_student.py time --name "$name" --repeats 15
  if ! grep -q "\"name\": \"$name\"" "$L/ladder.jsonl" 2>/dev/null; then
    "$PY" 05_distill/ladder_record.py record --rung B --seed 1 --steps "$STEPS" --name "$name" \
      --wall "$(cat "$wallf")" --arch "$arch" --frontend "$fe" --init "$init"
  fi
  "$PY" 05_distill/ladder_record.py frontier
}

# 1. inputs
DONE="$L/CACHE_FE_DONE_$(echo "$FES" | tr ',' '_')"
if [ ! -e "$DONE" ]; then
  echo "[chain] cache_fe V: start $(date +%T)"; "$ONNX_PY" 05_distill/cache_fe.py --rung V --frontends "$FES"
  echo "[chain] cache_fe B: start $(date +%T)"; "$ONNX_PY" 05_distill/cache_fe.py --rung B --frontends "$FES"
  touch "$DONE"
fi
# 2. runs
for r in $RUNS; do
  IFS=: read -r fe arch init <<< "$r"
  run_fe "$fe" "$arch" "${init:-}"
done
echo "[chain] all done"
