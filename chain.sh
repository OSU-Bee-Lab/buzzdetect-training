#!/usr/bin/env bash
# yam64h24: two d8 students on the hop-24 front end (rung C, WSD 7000-56000, stop rule), then the pad-free
# re-export of every judged branch (same weights), as exp/yam64h32 did. Run by ../yam64h32-s2/chain.sh.
set -e
cd "$(dirname "$0")"
PY=$(bash tools/python_path.sh)
C=classes=ins_buzz+ambient_rain+human
$PY 05_distill/main.py --rung C --seed 1 --wsd-max 56000 --wsd-stop 1.3 --runs "yam64h24:a0.50_d8:$C yam64h24:a0.25_d8:$C"
RUNS=$(cd 05_distill && $PY -c "
import dpaths as D
names = {r['name'] for r in D.read_log()}
print(' '.join(sorted(n for n in names if n.startswith('fe_C_yam64h24_') and not n.endswith('_mmt') and n + '_mmt' not in names)))" 2>/dev/null)
echo "[chain] re-export with mmt: $RUNS"
[ -z "$RUNS" ] || $PY 05_distill/reexport.py --fe-impl mmt --suffix _mmt --runs $RUNS
