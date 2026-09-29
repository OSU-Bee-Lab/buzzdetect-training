#!/bin/bash
# NOTE (2026-09-29): pre-generalization chain. Hard-codes the single-teacher layout under .local/distill and the old
# _fe cache; kept as the record of the runs it made. New work: 05_distill/main.py (README.md), same runs, any teacher.
# Resume chain_frontends.sh's default run list after chain 2 (WAIT_PID = its launch_job pid), so the one
# GPU job at a time rule holds. Finished stages skip, so this picks up at fe_B_fast32h16_a0.50_s1 export
# (chain 1 died there on the pad off-by-one fixed in frontends.py) and carries on down the list.
#   WAIT_PID=<pid> tools/launch_job.sh <log> -- bash 05_distill/chain_frontends3.sh
set -euo pipefail
WT=/home/luke/projects/buzzdetect-training/.claude/worktrees/distill-lite
cd "$WT"
if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
unset RUNS
bash 05_distill/chain_frontends.sh
