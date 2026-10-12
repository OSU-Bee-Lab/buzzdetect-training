#!/usr/bin/env bash
# yam64h16: random-weight speed of the new front end and its d8 trunks, pad-free export, before training;
# the yamnet and twofast32 twins are re-timed in the same session
set -e
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES='' $(bash tools/python_path.sh) 05_distill/bench_arch.py export --out /home/luke/projects/buzzdetect-training/05_distill/data/_shared/arch_mm --candidates fe@yam64h16+mmt a0.25_d8@yam64h16+mmt a0.375_d8@yam64h16+mmt a0.50_d8@yam64h16+mmt a0.25@yam64h16+mmt fe@yamnet+mmt a0.25_d8@yamnet+mmt a0.50_d8@yamnet+mmt a0.50_d8@twofast32+mmt a0.25_d8@twofast32+mmt
/home/luke/projects/buzzdetect/engine/.venv/bin/python3 05_distill/bench_arch.py time --seconds 200 --out /home/luke/projects/buzzdetect-training/05_distill/data/_shared/arch_mm --candidates fe@yam64h16+mmt a0.25_d8@yam64h16+mmt a0.375_d8@yam64h16+mmt a0.50_d8@yam64h16+mmt a0.25@yam64h16+mmt fe@yamnet+mmt a0.25_d8@yamnet+mmt a0.50_d8@yamnet+mmt a0.50_d8@twofast32+mmt a0.25_d8@twofast32+mmt
