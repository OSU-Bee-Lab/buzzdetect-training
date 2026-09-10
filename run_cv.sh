#!/usr/bin/env bash
set -x
P=/home/luke/anaconda3/envs/buzzdetect-train/bin/python
cd /home/luke/projects/buzzdetect-training/.local/worktrees/aves-mlp-head
$P -u 03_train/main.py --name aves_h0   --set medium --embedder aves --translation general --verbose -y
$P -u 03_train/main.py --name aves_h256 --set medium --embedder aves --translation general --hidden 256  --verbose -y
$P -u 03_train/main.py --name aves_h1024 --set medium --embedder aves --translation general --hidden 1024 --verbose -y
echo "ALL_DONE"
