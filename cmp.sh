#!/usr/bin/env bash
PY=$(bash tools/python_path.sh)
S=s1_c-buzz-rain-human_wsd56000_mmt
for a in a0.50_d8 a0.25_d8; do
  for c in yam64h32 yam64h16; do
    echo "=== $c vs yam64h24 $a"
    $PY tools/results.py fe_C_${c}_${a}_$S fe_C_yam64h24_${a}_$S
  done
done
echo "=== twofast32 a0.50_d8 vs yam64h24 a0.50_d8"
$PY tools/results.py fe_C_twofast32_a0.50_d8_$S fe_C_yam64h24_a0.50_d8_$S
