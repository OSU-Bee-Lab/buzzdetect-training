#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
D=/home/luke/projects/buzzdetect-training/05_distill/data/_shared/arch_mm/_micro2
CUDA_VISIBLE_DEVICES='' $(bash tools/python_path.sh) micro/make2.py $D
/home/luke/projects/buzzdetect/engine/.venv/bin/python3 micro/time.py $D 200
