#!/bin/bash
set -e
cd "$(dirname "$0")"

EMBEDDER=yamnet
SET=medium
TRANSLATION=general
EPOCHS=300
LS=0.2
LR=0.002

for DROPOUT in 0.0 0.1 0.2 0.3 0.4; do
    SLUG=$(echo "$DROPOUT" | tr -d '.')
    MODEL="exp_hp_drop${SLUG}_v1"
    echo "=== Training $MODEL (dropout=$DROPOUT) ==="
    conda run -n buzzdetect-train python 03_train/main.py \
        --model "$MODEL" --set "$SET" --embedder "$EMBEDDER" \
        --translation "$TRANSLATION" --epochs "$EPOCHS" \
        --dropout "$DROPOUT" --label-smoothing "$LS" --learning-rate "$LR"
    echo "=== Done: $MODEL ==="
done

echo "=== Dropout sweep complete ==="
