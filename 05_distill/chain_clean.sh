#!/bin/bash
# Retrain every student that trained on test-set audio, on packs rebuilt from the clean plan (2026-10-04).
# Packs without a fingerprint predated the 2026-10-02 SeeNote blacklist; quarantine.py moved the 64 affected
# runs to data/<teacher>/_contaminated_2026-10-02/ (manifest.json there has their args). FRONTENDS.md,
# "Update 2026-10-04". Steps, in order, on the one GPU (~45 h): a failed step is retried once (every stage
# resumes), then the chain moves on; re-running the chain redoes only what is missing.
#   0 fill the caches for the clean plan (its new-audio slices): teacher targets to rung C, fast mels to rung B;
#     then re-score V on the 11 clean fast32h16 rung-C runs (their packs were clean, their V pool was not)
#   1 rung-A repeats lad_A_s1/s2 (the noise floor: --wsd-stop's hit@K spread)
#   2 the 46 rung-B cosine students, one main.py each (frontier_svg.py's points)
#   3 WSD: rung-B 7k equivalence pair, yamnet a0.25 rung-B curve, yamnet a0.50 rung-C curve (rule on)
#   4 the rest of the old ladder: lad_B_s1, lad_B_a0.375_s1, lad_B_a0.50_d12_s1, lad_C_s1
#   5 proxy table and the human-facing plots
#
#   tools/launch_job.sh 05_distill/data/chain_clean.log -- bash 05_distill/chain_clean.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source tools/python_path.sh   # PY
export PYTHONUNBUFFERED=1
export DISTILL_TEACHER=v4-ft-ps-e60-moderate
L=05_distill/data/$DISTILL_TEACHER
ENGINE_PY=$(cd 05_distill && "$PY" -c "import dpaths as D; print(D.ENGINE_PY)")
BRH=classes=ins_buzz+ambient_rain+human
FAILED=()

attempt() {  # attempt <label> <cmd...>: run, retry once
  local label=$1; shift
  for n in 1 2; do
    echo "[cleanchain] $label: attempt $n start $(date +%T)"
    if "$@"; then echo "[cleanchain] $label: done $(date +%T)"; return 0; fi
    echo "[cleanchain] $label: attempt $n FAILED $(date +%T)"
  done
  FAILED+=("$label"); return 1
}
main() { "$PY" -u 05_distill/main.py --teacher "$DISTILL_TEACHER" "$@"; }
cos() { local rung=$1 seed=$2 spec=$3; attempt "cos $rung s$seed $spec" main --rung "$rung" --seed "$seed" --runs "$spec"; }

lad_run() {  # the old ladder runs (chain_ladder2.sh's stages, plus probe): lad_run <name> <rung> <seed> [<arch>]
  local name=$1 rung=$2 seed=$3 arch=${4:-} d=$L/runs/$1 m=$L/models/$1 ev=$L/eval/$1
  if [ ! -f "$d/TRAIN_DONE" ]; then
    local t0=$SECONDS
    (cd 05_distill && "$PY" -u distill_train.py --rung "$rung" --steps 7000 --batch 512 --seed "$seed" --name "$name" \
       ${arch:+--arch "$arch"} --loader mem --eval-every 2000) || return 1
    touch "$d/TRAIN_DONE"; echo $((SECONDS - t0)) > "$d/wall.txt"
  fi
  [ -f "$m/model.onnx" ] || (cd 05_distill && "$PY" -u export_student.py export --run "$name" --name "$name") || return 1
  [ -f "$ev/folds_sx.csv" ] || (cd 05_distill && "$PY" -u eval_folds.py run --check-labels --onnx "../$m/model.onnx" --out "../$ev") || return 1
  [ -f "$ev/probe/probe.json" ] || (cd 05_distill && "$PY" -u eval_folds.py probe --onnx "../$m/model.onnx" --out "../$ev") || return 1
  [ -f "$m/speed_200.json" ] || (cd 05_distill && "$ENGINE_PY" export_student.py time --name "$name" --repeats 15) || return 1
  local wall; wall=$(cat "$d/wall.txt") || return 1    # read before the cd: $d is relative to the repo root
  grep -q "\"name\": \"$name\"" "$L/ladder.jsonl" || (cd 05_distill && "$PY" ladder_record.py record --rung "$rung" \
    --seed "$seed" --steps 7000 --name "$name" --wall "$wall" ${arch:+--arch "$arch"}) || return 1
}
lad() { attempt "$1" lad_run "$@"; }

attempt cache-targets-ABC main --rung C --until cache --runs yamnet:a0.50
attempt cache-mels-B main --rung B --until cache_fe --runs \
  "fast32:a0.50 fast32h16:a0.50 fast32h16lo:a0.50 fast32h32:a0.50 fast32lo:a0.50 lo32:a0.50 two32:a0.50 twofast32:a0.50"
attempt rescore-clean-C "$PY" -u 05_distill/backfill_hitk.py --rescore --names \
fe_C_fast32h16_a0.25_s1_c-buzz-rain-human_wsd14000,fe_C_fast32h16_a0.25_s1_c-buzz-rain-human_wsd28000,fe_C_fast32h16_a0.25_s1_c-buzz-rain-human_wsd56000,fe_C_fast32h16_a0.25_s1_c-buzz-rain-human_wsd7000,fe_C_fast32h16_a0.50_s1_c-buzz-rain-human_wsd112000,fe_C_fast32h16_a0.50_s1_c-buzz-rain-human_wsd14000,fe_C_fast32h16_a0.50_s1_c-buzz-rain-human_wsd28000,fe_C_fast32h16_a0.50_s1_c-buzz-rain-human_wsd56000,fe_C_fast32h16_a0.50_s1_c-buzz-rain-human_wsd7000

lad lad_A_s1 A 1
lad lad_A_s2 A 2

cos B 1 fast32:a0.25:classes=ins_buzz
cos B 1 fast32:a0.25:classes=ins_buzz+ambient_rain+human
cos B 1 fast32:a0.375:classes=ins_buzz
cos B 1 fast32:a0.375:classes=ins_buzz+ambient_rain+human
cos B 1 fast32:a0.50
cos B 1 fast32:a0.50:classes=ins_buzz
cos B 1 fast32:a0.50:classes=ins_buzz+ambient_rain+human
cos B 1 fast32h16:a0.25
cos B 1 fast32h16:a0.25:classes=ins_buzz
cos B 1 fast32h16:a0.25:classes=ins_buzz+ambient_rain+human
cos B 1 fast32h16:a0.25:classes=ins_buzz+ambient_rain+human:lam=0
cos B 1 fast32h16:a0.25:lam=0
cos B 1 fast32h16:a0.375
cos B 1 fast32h16:a0.375:classes=ins_buzz
cos B 1 fast32h16:a0.375:classes=ins_buzz+ambient_rain+human
cos B 2 fast32h16:a0.375:classes=ins_buzz+ambient_rain+human
cos B 1 fast32h16:a0.50
cos B 1 fast32h16:a0.50:classes=ins_buzz
cos B 1 fast32h16:a0.50:classes=ins_buzz+ambient_rain+human
cos B 1 fast32h16lo:a0.50
cos B 1 fast32h32:a0.25
cos B 1 fast32h32:a0.50
cos B 1 fast32lo:a0.25:classes=ins_buzz
cos B 1 fast32lo:a0.25:classes=ins_buzz+ambient_rain+human
cos B 1 fast32lo:a0.375:classes=ins_buzz
cos B 1 fast32lo:a0.375:classes=ins_buzz+ambient_rain+human
cos B 1 fast32lo:a0.50
cos B 1 fast32lo:a0.50:classes=ins_buzz
cos B 1 fast32lo:a0.50:classes=ins_buzz+ambient_rain+human
cos B 1 lo32:a0.50
cos B 1 two32:a0.50
cos B 1 twofast32:a0.25:classes=ins_buzz
cos B 1 twofast32:a0.25:classes=ins_buzz+ambient_rain+human
cos B 1 twofast32:a0.375
cos B 1 twofast32:a0.375:classes=ins_buzz
cos B 1 twofast32:a0.375:classes=ins_buzz+ambient_rain+human
cos B 1 twofast32:a0.50
cos B 1 twofast32:a0.50:classes=ins_buzz
cos B 1 twofast32:a0.50:classes=ins_buzz+ambient_rain+human
cos B 1 yamnet:a0.25:select:classes=ins_buzz
cos B 1 yamnet:a0.25:select:classes=ins_buzz+ambient_rain+human
cos B 1 yamnet:a0.375:select:classes=ins_buzz
cos B 1 yamnet:a0.375:select:classes=ins_buzz+ambient_rain+human
cos B 1 yamnet:a0.50:select
cos B 1 yamnet:a0.50:select:classes=ins_buzz
cos B 1 yamnet:a0.50:select:classes=ins_buzz+ambient_rain+human

STOP=$(cd 05_distill && "$PY" ladder_record.py spread) || { echo "[cleanchain] no hit@K spread; WSD steps use 1.85"; STOP=1.85; }
echo "[cleanchain] --wsd-stop $STOP (rung-A hit@K spread)"
attempt wsd-equivalence main --rung B --wsd 7000 --runs "yamnet:a0.50:select:$BRH fast32h16:a0.50:$BRH"
attempt wsd-yamnet-a0.25-B main --rung B --wsd 7000,14000,28000,70000 --wsd-stop "$STOP" --runs "yamnet:a0.25:select:$BRH"
attempt wsd-yamnet-a0.50-C main --rung C --wsd-max 56000 --wsd-stop "$STOP" --runs "yamnet:a0.50:select:$BRH"

lad lad_B_s1 B 1
lad lad_B_a0.375_s1 B 1 a0.375
lad lad_B_a0.50_d12_s1 B 1 a0.50_d12
lad lad_C_s1 C 1 a0.50

(cd 05_distill && "$PY" ladder_record.py proxy) || true
(cd 05_distill && "$PY" ladder_record.py frontier) || true
"$PY" tools/human/frontier_svg.py || true
"$PY" tools/human/frontier_html.py || true
"$PY" tools/human/wsd_svg.py --tol "$STOP" || true
echo "[cleanchain] failed steps: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
