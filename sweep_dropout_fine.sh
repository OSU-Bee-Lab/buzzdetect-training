#!/bin/bash
set -e
cd "$(dirname "$0")"

EMBEDDER=yamnet
SET=medium
TRANSLATION=general
EPOCHS=300
LS=0.2
LR=0.002

for DROPOUT in 0.025 0.050 0.075 0.125 0.150 0.175 0.225 0.250 0.275 0.325 0.350 0.375; do
    SLUG=$(printf "%.3f" "$DROPOUT" | tr -d '.')
    MODEL="exp_hp_drop${SLUG}_v1"
    echo "=== Training $MODEL (dropout=$DROPOUT) ==="
    conda run -n buzzdetect-train python 03_train/main.py \
        --model "$MODEL" --set "$SET" --embedder "$EMBEDDER" \
        --translation "$TRANSLATION" --epochs "$EPOCHS" \
        --dropout "$DROPOUT" --label-smoothing "$LS" --learning-rate "$LR"
    echo "=== Done: $MODEL ==="
done

echo "=== Fine dropout sweep complete ==="
