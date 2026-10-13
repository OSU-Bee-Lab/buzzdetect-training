#!/usr/bin/env bash
# the 3-draw mean against each member, and the new draws against the first
cd "$(dirname "$0")"
PY=$(bash tools/python_path.sh)
for m in v4-ft-ps-e60-moderate v4-ft-ps-e60-moderate-r2 v4-ft-ps-e60-moderate-r3; do
  $PY tools/results.py $m ens3-moderate
done
$PY tools/results.py v4-ft-ps-e60-moderate v4-ft-ps-e60-moderate-r2
$PY tools/results.py v4-ft-ps-e60-moderate v4-ft-ps-e60-moderate-r3
