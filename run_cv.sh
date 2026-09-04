#!/usr/bin/env bash
# Restart wrapper for the trunk-ft-restore-sens CV.
#
# Per-fold OOM between rotations was routine on this 4 GB GPU during the
# original trunk-ft runs. 03_train resumes from disk (train_utils.can_write()
# skips any fold dir that already holds a config_model.json), so the fix is to
# relaunch main.py until folds_sx.csv exists. Give up after MAX tries so a
# genuine crash loop doesn't spin forever.
set -u
NAME=trunk_ft_restore_sens
MAX=40
i=0
while [ ! -f "models/$NAME/folds_sx.csv" ] && [ "$i" -lt "$MAX" ]; do
    i=$((i + 1))
    echo "=== attempt $i $(date +%F_%T) ==="
    /home/luke/anaconda3/envs/buzzdetect-train/bin/python -u 03_train/main.py \
        --name "$NAME" --set medium --embedder yamnet_trunk --translation general \
        --batch 1024 --lr-backbone 1e-5 --lr-head 2e-4 \
        --patience 50 --epochs 400 --verbose -y
    echo "=== attempt $i exited rc=$? ==="
done
if [ -f "models/$NAME/folds_sx.csv" ]; then
    echo "=== CV COMPLETE $(date +%F_%T) ==="
else
    echo "=== GAVE UP after $i attempts $(date +%F_%T) ==="
fi
