#!/usr/bin/env bash
set -e
cd "$(dirname "$(realpath "$0")")"
PY=$(bash tools/python_path.sh)
$PY 05_distill/main.py --rung C --seed 1 --wsd-max 56000 --wsd-stop 1.3 \
  --runs "b64w16h16:a0.25_d8:classes=ins_buzz+ambient_rain+human b64w16h16:a0.50_d8:classes=ins_buzz+ambient_rain+human"
RUNS=$(cd 05_distill && $PY -c "
import dpaths as D
names = {r['name'] for r in D.read_log()}
p = 'fe_C_b64w16h16_'
print(' '.join(sorted(n for n in names if n.startswith(p) and not n.endswith('_mmt') and n + '_mmt' not in names)))" 2>/dev/null)
echo "[chain] re-export with mmt: $RUNS"
[ -z "$RUNS" ] || $PY 05_distill/reexport.py --fe-impl mmt --suffix _mmt --runs $RUNS
