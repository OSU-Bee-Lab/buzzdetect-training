#!/usr/bin/env bash
# batch 39 resume: chain 1 died on pauc-l01's missing train_utils import (fixed in this worktree);
# wait for chain 2 (pid 297592, the GPU) to exit, then rerun chain 1 (mixstyle-xfold is skipped as done).
while kill -0 297592 2>/dev/null; do sleep 60; done
echo "[chain] chain 2 ended"
exec bash /home/luke/projects/buzzdetect-training/.local/worktrees/mixstyle-xfold/chain_b39.sh
