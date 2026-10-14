#!/usr/bin/env bash
# d1-ens-teacher, step 2 (after chain_ship.sh and tools_exp/build_ens_teacher.py): distil twofast32:a0.50_d8
# (rung C, WSD 7000-56000, stop rule, buzz+rain+human, seed 1) from the 3-draw ensemble teacher, then the
# pad-free mmt re-export of every judged branch, as exp/yam64h16 did. Resumes on rerun.
set -e
cd "$(dirname "$(realpath "$0")")"
PY=$(bash tools/python_path.sh)
export DISTILL_TEACHER=v4-ft-ps-e60-moderate-ens3
C=classes=ins_buzz+ambient_rain+human
$PY 05_distill/main.py --teacher $DISTILL_TEACHER --prefix ens3 --rung C --seed 1 --wsd-max 56000 --wsd-stop 1.3 \
  --runs "twofast32:a0.50_d8:$C"
RUNS=$(cd 05_distill && $PY -c "
import dpaths as D
names = {r['name'] for r in D.read_log()}
print(' '.join(sorted(n for n in names if n.startswith('ens3_C_twofast32_') and not n.endswith('_mmt') and n + '_mmt' not in names)))" 2>/dev/null)
echo "[chain] re-export with mmt: $RUNS"
[ -z "$RUNS" ] || $PY 05_distill/reexport.py --fe-impl mmt --suffix _mmt --runs $RUNS
