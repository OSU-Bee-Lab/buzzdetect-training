#!/usr/bin/env bash
# twofast32-d8-s2: seed-2 repeats of the two twofast32 d8 students (rung C, the budgets seed 1 recorded),
# then the pad-free re-export of every judged branch (same weights), as exp/twofast32-d8 did for seed 1.
set -e
cd "$(dirname "$0")"
PY=$(bash tools/python_path.sh)
C=classes=ins_buzz+ambient_rain+human
$PY 05_distill/main.py --rung C --seed 2 --wsd 7000,14000,28000,56000 --runs "twofast32:a0.50_d8:$C twofast32:a0.25_d8:$C"
RUNS=$(cd 05_distill && $PY -c "
import dpaths as D
names = {r['name'] for r in D.read_log()}
print(' '.join(sorted(n for n in names if n.startswith('fe_C_twofast32_a0.') and '_d8_s2_' in n and not n.endswith('_mmt') and n + '_mmt' not in names)))" 2>/dev/null)
echo "[chain] re-export with mmt: $RUNS"
[ -z "$RUNS" ] || $PY 05_distill/reexport.py --fe-impl mmt --suffix _mmt --runs $RUNS
