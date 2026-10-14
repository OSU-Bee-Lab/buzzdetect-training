#!/usr/bin/env bash
# yam64h32: bench, then two d8 students on the new front end (rung C, WSD 7000-56000, stop rule), then the pad-free
# re-export of every judged branch (same weights), as exp/twofast32-d8 did.
set -e
cd "$(dirname "$0")"
PY=$(bash tools/python_path.sh)
bash bench.sh > bench.log 2>&1 || echo "[chain] bench failed, see bench.log"
C=classes=ins_buzz+ambient_rain+human
$PY 05_distill/main.py --rung C --seed 1 --wsd-max 56000 --wsd-stop 1.3 --runs "yam64h32:a0.25_d8:$C yam64h32:a0.50_d8:$C"
RUNS=$(cd 05_distill && $PY -c "
import dpaths as D
names = {r['name'] for r in D.read_log()}
print(' '.join(sorted(n for n in names if n.startswith('fe_C_yam64h32_') and not n.endswith('_mmt') and n + '_mmt' not in names)))" 2>/dev/null)
echo "[chain] re-export with mmt: $RUNS"
[ -z "$RUNS" ] || $PY 05_distill/reexport.py --fe-impl mmt --suffix _mmt --runs $RUNS
