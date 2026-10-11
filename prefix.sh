#!/usr/bin/env bash
for g in "fe@fast32h16+mm" "fe@fast32h16" "a0.25@fast32h16+mm"; do echo "##### $g"
  CUDA_VISIBLE_DEVICES='' $(bash tools/python_path.sh) 05_distill/prefix_bench.py cut "/home/luke/projects/buzzdetect-training/05_distill/data/_shared/arch_mm/$g/model.onnx" 2>&1 | tail -1
  /home/luke/projects/buzzdetect/engine/.venv/bin/python3 05_distill/prefix_bench.py time "/home/luke/projects/buzzdetect-training/05_distill/data/_shared/arch_mm/$g/model.onnx" 2>&1 | grep -v "^20[0-9][0-9]-\|Warning\|warning"; done
