#!/usr/bin/env bash
set -e
P=$(bash tools/python_path.sh)
R="yam64h32:a0.25_d8:classes=ins_buzz+ambient_rain+human"
$P 05_distill/main.py --rung C --runs "$R" --wsd 7000,14000,28000,56000,112000 --wsd-eval 112000
$P 05_distill/reexport.py --fe-impl mmt --suffix _mmt --runs fe_C_yam64h32_a0.25_d8_s1_c-buzz-rain-human_wsd112000
