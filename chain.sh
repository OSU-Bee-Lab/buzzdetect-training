#!/usr/bin/env bash
# nonuniform-width: twofast32 a0.25_d8 trunks with wide late layers (rung C, WSD 7000-56000, stop rule,
# buzz+rain+human, seed 1), then the pad-free mmt re-export of every judged branch; then
# ../nonuniform-width-yam64h32/chain.sh (the same pair on yam64h32), in one job. Resumes on rerun.
set -e
cd "$(dirname "$(realpath "$0")")"
PY=$(bash tools/python_path.sh)
C=classes=ins_buzz+ambient_rain+human
FE=${1:-twofast32}
$PY 05_distill/main.py --rung C --seed 1 --wsd-max 56000 --wsd-stop 1.3 \
  --runs "$FE:a0.25_d8_l5x0.50:$C $FE:a0.25_d8_l7x1.0:$C"
RUNS=$(cd 05_distill && FE=$FE $PY -c "
import os, dpaths as D
names = {r['name'] for r in D.read_log()}
p = 'fe_C_' + os.environ['FE'] + '_a0.25_d8_l'
print(' '.join(sorted(n for n in names if n.startswith(p) and not n.endswith('_mmt') and n + '_mmt' not in names)))" 2>/dev/null)
echo "[chain] re-export with mmt: $RUNS"
[ -z "$RUNS" ] || $PY 05_distill/reexport.py --fe-impl mmt --suffix _mmt --runs $RUNS
if [ "$FE" = twofast32 ] && [ -x ../nonuniform-width-yam64h32/chain.sh ]; then
  echo "[chain] next: nonuniform-width-yam64h32"
  ../nonuniform-width-yam64h32/chain.sh yam64h32
fi
