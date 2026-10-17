#!/usr/bin/env bash
set -e
cd "$(dirname "$(realpath "$0")")"
PY=$(bash tools/python_path.sh)
$PY 05_distill/main.py --rung B --seed 2 --wsd 7000,14000,28000,56000 \
  --runs "fast32h16:a0.50:classes=ins_buzz+ambient_rain+human"
