#!/bin/bash
set -e
cd "$(dirname "$0")"

EMBEDDER=yamnet
SET=medium
TRANSLATION=general
EPOCHS=300
DROPOUT=0.4
LR=0.002

for LS in 0.10 0.15 0.20 0.25; do
    SLUG=$(echo "$LS" | tr -d '.')
    MODEL="exp_hp_ls${SLUG}_v1"
    echo "=== Training $MODEL (label_smoothing=$LS) ==="
    conda run -n buzzdetect-train python 03_train/main.py \
        --model "$MODEL" --set "$SET" --embedder "$EMBEDDER" \
        --translation "$TRANSLATION" --epochs "$EPOCHS" \
        --dropout "$DROPOUT" --label-smoothing "$LS" --learning-rate "$LR"
    echo "=== Done: $MODEL ==="
done

echo "=== Label smooth sweep complete ==="
