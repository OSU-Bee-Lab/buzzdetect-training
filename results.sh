#!/usr/bin/env bash
cd "$(dirname "$0")"
PY=$(bash tools/python_path.sh)
S=_s1_c-buzz-rain-human_wsd56000
$PY tools/results.py fe_C_twofast32_a0.50$S fe_C_twofast32_a0.50_d8$S
$PY tools/results.py fe_C_twofast32_a0.50$S fe_C_twofast32_a0.25_d8$S
$PY tools/results.py fe_C_fast32h16_a0.50_d8$S fe_C_twofast32_a0.25_d8$S
$PY tools/results.py fe_C_fast32h16_a0.75_d8$S fe_C_twofast32_a0.50_d8$S
