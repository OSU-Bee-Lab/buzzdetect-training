#!/bin/bash
set -e
cd "$(dirname "$0")"

EMBEDDER=yamnet
SET=medium
TRANSLATION=general
EPOCHS=300
DROPOUT=0.4
LS=0.20  # updated after label smooth sweep if a better value is found

for LR in 0.001 0.002 0.003; do
    SLUG=$(echo "$LR" | tr -d '.')
    MODEL="exp_hp_lr${SLUG}_v1"
    echo "=== Training $MODEL (lr=$LR) ==="
    conda run -n buzzdetect-train python 03_train/main.py \
        --model "$MODEL" --set "$SET" --embedder "$EMBEDDER" \
        --translation "$TRANSLATION" --epochs "$EPOCHS" \
        --dropout "$DROPOUT" --label-smoothing "$LS" --learning-rate "$LR"
    echo "=== Done: $MODEL ==="
done

echo "=== LR sweep complete ==="
