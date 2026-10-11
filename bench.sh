#!/usr/bin/env bash
# twofast32-d8: random-weight speed of trunk cuts on twofast32, pad-free export, before training
set -e
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES='' $(bash tools/python_path.sh) 05_distill/bench_arch.py export --out /home/luke/projects/buzzdetect-training/05_distill/data/_shared/arch_mm --candidates fe@twofast32+mmt a0.50@twofast32+mmt a0.50_d10@twofast32+mmt a0.50_d8@twofast32+mmt a0.75_d8@twofast32+mmt a0.375_d8@twofast32+mmt a0.25_d8@twofast32+mmt a0.25@twofast32+mmt a0.50_d8@fast32h16+mmt a0.75_d8@fast32h16+mmt a0.25@fast32h16+mmt
/home/luke/projects/buzzdetect/engine/.venv/bin/python3 05_distill/bench_arch.py time --seconds 200 --out /home/luke/projects/buzzdetect-training/05_distill/data/_shared/arch_mm --candidates fe@twofast32+mmt a0.50@twofast32+mmt a0.50_d10@twofast32+mmt a0.50_d8@twofast32+mmt a0.75_d8@twofast32+mmt a0.375_d8@twofast32+mmt a0.25_d8@twofast32+mmt a0.25@twofast32+mmt a0.50_d8@fast32h16+mmt a0.75_d8@fast32h16+mmt a0.25@fast32h16+mmt
