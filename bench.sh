#!/usr/bin/env bash
# matmul-stft bench: export random-weight graphs (train env, GPU hidden), then time them at 200 s and 20 s
set -e
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES='' $(bash tools/python_path.sh) 05_distill/bench_arch.py export --out /home/luke/projects/buzzdetect-training/05_distill/data/_shared/arch_mm --candidates floor fe@fast32h16 fe@fast32h16+conv fe@fast32h16+mm a0.25@fast32h16 a0.25@fast32h16+mm a0.50_d8@fast32h16 a0.50_d8@fast32h16+mm fe@fast32h32 fe@fast32h32+mm fe@twofast32 fe@twofast32+conv fe@twofast32+mm a0.50@twofast32 a0.50@twofast32+mm fe@fast32 fe@yamnet+conv
E=/home/luke/projects/buzzdetect/engine/.venv/bin/python3
$E 05_distill/bench_arch.py time --seconds 200 --out /home/luke/projects/buzzdetect-training/05_distill/data/_shared/arch_mm --candidates floor fe@fast32h16 fe@fast32h16+conv fe@fast32h16+mm a0.25@fast32h16 a0.25@fast32h16+mm a0.50_d8@fast32h16 a0.50_d8@fast32h16+mm fe@fast32h32 fe@fast32h32+mm fe@twofast32 fe@twofast32+conv fe@twofast32+mm a0.50@twofast32 a0.50@twofast32+mm fe@fast32 fe@yamnet+conv
echo "=== 20 s"
$E 05_distill/bench_arch.py time --seconds 20 --out /home/luke/projects/buzzdetect-training/05_distill/data/_shared/arch_mm --candidates floor fe@fast32h16 fe@fast32h16+conv fe@fast32h16+mm a0.25@fast32h16 a0.25@fast32h16+mm a0.50_d8@fast32h16 a0.50_d8@fast32h16+mm fe@fast32h32 fe@fast32h32+mm fe@twofast32 fe@twofast32+conv fe@twofast32+mm a0.50@twofast32 a0.50@twofast32+mm fe@fast32 fe@yamnet+conv
echo "=== 200 s again"
$E 05_distill/bench_arch.py time --seconds 200 --out /home/luke/projects/buzzdetect-training/05_distill/data/_shared/arch_mm --candidates floor fe@fast32h16 fe@fast32h16+conv fe@fast32h16+mm a0.25@fast32h16 a0.25@fast32h16+mm a0.50_d8@fast32h16 a0.50_d8@fast32h16+mm fe@fast32h32 fe@fast32h32+mm fe@twofast32 fe@twofast32+conv fe@twofast32+mm a0.50@twofast32 a0.50@twofast32+mm fe@fast32 fe@yamnet+conv
