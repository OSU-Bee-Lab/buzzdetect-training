#!/usr/bin/env bash
set -e
cd "$(dirname "$(realpath "$0")")"
PY=$(bash tools/python_path.sh)
$PY 05_distill/cache_fe.py --rung C --frontends yam64h16sp2
$PY 05_distill/cache_fe.py --rung V --frontends yam64h16sp2
$PY 05_distill/main.py --rung C --seed 1 --wsd-max 56000 --wsd-stop 1.3 \
  --runs "yam64h16sp2:a0.25_d8:classes=ins_buzz+ambient_rain+human yam64h16sp2:a0.50_d8:classes=ins_buzz+ambient_rain+human"
RUNS=$(cd 05_distill && $PY -c "
import dpaths as D
names = {r['name'] for r in D.read_log()}
p = 'fe_C_yam64h16sp2_'
print(' '.join(sorted(n for n in names if n.startswith(p) and not n.endswith('_mmtz') and n + '_mmtz' not in names)))" 2>/dev/null)
echo "[chain] re-export with mmtz: $RUNS"
[ -z "$RUNS" ] || (cd 05_distill && $PY reexport.py --fe-impl mmtz --suffix _mmtz --runs $RUNS)
# same-session speed against the full-band twins (logged speeds drift 5-10% between sessions)
T=$PWD/.local/bench && mkdir -p $T && N=""
for a in a0.25_d8 a0.50_d8; do for fe in yam64h16 yam64h16sp2; do
  n=fe_C_${fe}_${a}_s1_c-buzz-rain-human_wsd56000_mmtz; ln -sfn /home/luke/projects/buzzdetect-training/05_distill/data/v4-ft-ps-e60-moderate/models/$n $T/$n; N="$N $n"; done; done
for n in fe_C_twofast32_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt fe_C_yam64h32_a0.25_d8_l5x0.50_s1_c-buzz-rain-human_wsd56000_mmt fe_C_yam64h32_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt; do ln -sfn /home/luke/projects/buzzdetect-training/05_distill/data/v4-ft-ps-e60-moderate/models/$n $T/$n; N="$N $n"; done
for r in 1 2; do (cd 05_distill && /home/luke/projects/buzzdetect/engine/.venv/bin/python3 bench_arch.py time --seconds 200 --repeats 15 --out $T --candidates$N); cp $T/results.json $T/results_$r.json; done
