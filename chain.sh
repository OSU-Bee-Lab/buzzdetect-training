#!/usr/bin/env bash
# yam64h32-s2: seed-2 repeat of yam64h32:a0.25_d8 (rung C, the budgets seed 1 recorded), then the pad-free
# re-export of its branches. Then runs ../yam64h24/chain.sh (batch 33's next experiment) in the same job.
set -e
cd "$(dirname "$0")"
PY=$(bash tools/python_path.sh)
C=classes=ins_buzz+ambient_rain+human
$PY 05_distill/main.py --rung C --seed 2 --wsd 7000,14000,28000,56000 --runs "yam64h32:a0.25_d8:$C"
RUNS=$(cd 05_distill && $PY -c "
import dpaths as D
names = {r['name'] for r in D.read_log()}
print(' '.join(sorted(n for n in names if n.startswith('fe_C_yam64h32_a0.25_d8_s2_') and not n.endswith('_mmt') and n + '_mmt' not in names)))" 2>/dev/null)
echo "[chain] re-export with mmt: $RUNS"
[ -z "$RUNS" ] || $PY 05_distill/reexport.py --fe-impl mmt --suffix _mmt --runs $RUNS
echo "[chain] yam64h32-s2 done; starting yam64h24"
bash ../yam64h24/chain.sh
