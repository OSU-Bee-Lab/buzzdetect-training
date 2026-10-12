#!/usr/bin/env bash
# each yam64h16 student against its 10 ms-hop twin (frame-rate cost) and against the frontier student it must beat
cd "$(dirname "$0")"
PY=$(bash tools/python_path.sh)
T=_c-buzz-rain-human_wsd56000
$PY tools/results.py fe_C_yamnet_a0.25_d8_s1_select$T fe_C_yam64h16_a0.25_d8_s1$T
$PY tools/results.py fe_C_yamnet_a0.50_d8_s1_select$T fe_C_yam64h16_a0.50_d8_s1$T
$PY tools/results.py fe_C_twofast32_a0.50_d8_s1$T fe_C_yam64h16_a0.25_d8_s1$T
$PY tools/results.py fe_C_twofast32_a0.50_d8_s1$T fe_C_yam64h16_a0.50_d8_s1$T
