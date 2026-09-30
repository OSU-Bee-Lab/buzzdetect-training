#!/usr/bin/env bash
# Batch-16 chain: four one-lever CVs on yamnet_trunk_pitchshift_depth12, each in its own worktree.
# tools/launch_job.sh chain16.log -- .local/worktrees/ps-gmp/run_chain.sh   (rerun resumes)
R=/home/luke/projects/buzzdetect-training/.local/worktrees
PY=/home/luke/anaconda3/envs/buzzdetect-train/bin/python
export BUZZDETECT_CHUNK_FRAMES=48 TRUNK_FP16=1 TRUNK_LR_HEAD=2e-4 TRUNK_LR_BACKBONE=1e-5 TRUNK_BATCH=1024
run() { # slug env
  echo "[chain] $(date '+%m-%d %H:%M') $1 ($2)"
  cd $R/$1 || return 1
  for i in 1 2 3 4 5 6; do
    env $2 $PY -u 03_train/main.py --name $1 --set medium --embedder yamnet_trunk_pitchshift_depth12 \
      --translation general --epochs 30 -y --verbose && return 0
    echo "[chain] retry $i $1"; sleep 5
  done; return 1
}
run ps-gmp TRUNK_GMP=1
run ps-featdrop TRUNK_SPDROP=0.2
run ps-bntrain TRUNK_BN_TRAIN=1
run ps-nosmooth TRAIN_LS=0
echo "[chain] CHAIN-DONE"
