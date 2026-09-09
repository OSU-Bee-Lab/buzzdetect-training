#!/usr/bin/env bash
# Perch v2 needs TF >= 2.21 (its SavedModel is a JAX export whose StableHLO
# artifact TF 2.16.2 cannot deserialize), but the pipeline is pinned to
# TF 2.16.2 and every model this era was trained under it. So the two stages
# run under different interpreters:
#
#   stage 2 (extraction)  -> .local/venv-perch-extract, TF 2.21, runs Perch
#   stage 3 (training)    -> the pinned conda env, TF 2.16.2, unchanged
#
# That split is safe because stage 3 never runs the embedder: it loads it with
# initialize=False purely to read n_embeddings, and reads the pickled
# embeddings off disk. Training therefore stays byte-for-byte comparable with
# every other model in log.jsonl.
set -euo pipefail

SET=${SET:-medium}
NAME=${NAME:-perch_probe}
WT=/home/luke/projects/buzzdetect-training/.local/worktrees/perch-probe
PY_EXTRACT=/home/luke/projects/buzzdetect-training/.local/venv-perch-extract/bin/python
PY_TRAIN=/home/luke/anaconda3/envs/buzzdetect-train/bin/python

cd "$WT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""   # 4 GB GPU OOMs; CPU ~= GPU for the probe
export MALLOC_ARENA_MAX=2

echo "=== 02 extract ($SET, perch) === $(date -Is)"
# 32 frames/pass: Perch's forward pass allocates ~1.2 GB of intermediates at
# 48, and the default 300 would want ~8 GB. Two CPU workers at 32 sit around
# 5 GB total, well inside this box's 20 GB.
BUZZDETECT_CHUNK_FRAMES=${CHUNK:-32} "$PY_EXTRACT" -u 02_set/main.py \
  --set "$SET" --embedder perch --workers "${WORKERS:-2}" --verbose

echo "=== 03 train ($NAME) === $(date -Is)"
"$PY_TRAIN" -u 03_train/main.py \
  --name "$NAME" --set "$SET" --embedder perch --translation general --verbose -y

echo "=== done === $(date -Is)"
