#!/usr/bin/env bash
# twofast32-d8: distil the two d8 trunks on twofast32 (rung C, WSD to 56000), then re-export every judged
# branch with the pad-free front end (same weights) so its speed is on the frontier's current axis.
set -e
cd "$(dirname "$0")"
PY=$(bash tools/python_path.sh)
C=classes=ins_buzz+ambient_rain+human
$PY 05_distill/main.py --rung C --runs "twofast32:a0.50_d8:$C twofast32:a0.25_d8:$C" --wsd-max 56000 --wsd-stop 1.3
RUNS=$(cd 05_distill && $PY -c "
import dpaths as D
names = {r['name'] for r in D.read_log()}
print(' '.join(sorted(n for n in names if n.startswith('fe_C_twofast32_a0.') and '_d8_' in n and not n.endswith('_mmt') and n + '_mmt' not in names)))" 2>/dev/null)
echo "[chain] re-export with mmt: $RUNS"
[ -z "$RUNS" ] || $PY 05_distill/reexport.py --fe-impl mmt --suffix _mmt --runs $RUNS
