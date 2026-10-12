#!/usr/bin/env bash
# seed 2 against seed 1 of the same student, and against the bar each seed-1 gain was read over
cd "$(dirname "$0")"
PY=$(bash tools/python_path.sh)
T=_c-buzz-rain-human_wsd56000
$PY tools/results.py fe_C_twofast32_a0.50_d8_s1$T fe_C_twofast32_a0.50_d8_s2$T
$PY tools/results.py fe_C_twofast32_a0.25_d8_s1$T fe_C_twofast32_a0.25_d8_s2$T
$PY tools/results.py fe_C_fast32h16_a0.75_d8_s1$T fe_C_twofast32_a0.50_d8_s2$T
$PY tools/results.py fe_C_fast32h16_a0.50_d8_s1$T fe_C_twofast32_a0.25_d8_s2$T
