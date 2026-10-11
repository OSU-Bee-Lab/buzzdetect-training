#!/usr/bin/env bash
# matmul-stft: does the Pad cost show in the deployed graphs too? Prefix timing, 200 s, read-only on the engine's models
cd "$(dirname "$0")"
for pair in "yamnet:/home/luke/projects/buzzdetect/engine/models/.archive/yamnet_large_general/model.onnx" "teacher:/home/luke/projects/buzzdetect/engine/models/v4-ft-ps-e60-moderate/model.onnx"; do
  n=${pair%%:*}; p=${pair#*:}; echo "##### $n $p"
  CUDA_VISIBLE_DEVICES='' $(bash tools/python_path.sh) 05_distill/prefix_bench.py cut "$p" --out /home/luke/projects/buzzdetect-training/05_distill/data/_shared/arch_mm/_prefix_$n 2>&1 | tail -1
  /home/luke/projects/buzzdetect/engine/.venv/bin/python3 05_distill/prefix_bench.py time "$p" --out /home/luke/projects/buzzdetect-training/05_distill/data/_shared/arch_mm/_prefix_$n 2>&1 | grep -v "^20[0-9][0-9]-\|arning\|Removing\|graph.cc"
done
