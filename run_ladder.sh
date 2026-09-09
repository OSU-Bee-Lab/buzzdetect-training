#!/usr/bin/env bash
# Width ladder for shared-trunk-head. Sequential: 3 frozen-probe CVs, ~9 min each.
set -u
PY=/home/luke/anaconda3/envs/buzzdetect-train/bin/python
cd /home/luke/projects/buzzdetect-training/.local/worktrees/shared-trunk-head
for h in 64 256 1024; do
  echo "=== ladder: hidden=$h starting $(date -Is) ==="
  CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 "$PY" -u 03_train/main.py \
    --name hidden_$h --set medium --embedder yamnet --translation general \
    --hidden $h --verbose -y --no-surprisal
  echo "=== ladder: hidden=$h finished $(date -Is) rc=$? ==="
done
echo "=== LADDER COMPLETE $(date -Is) ==="
