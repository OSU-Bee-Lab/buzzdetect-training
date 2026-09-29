#!/bin/bash
# Second front-end chain: band-placement twins of fast32 / fast32h16 (same window, hop and bands, so the
# same speed; only the band range moves from 125-7500 Hz to 100-2500 Hz). Waits for chain_frontends.sh
# (its launch_job pid, WAIT_PID) so the one GPU job at a time rule holds, then runs it with a custom list.
#   WAIT_PID=<pid> tools/launch_job.sh <log> -- bash 05_distill/chain_frontends2.sh
set -euo pipefail
WT=/home/luke/projects/buzzdetect-training/.claude/worktrees/distill-lite
cd "$WT"
if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
export RUNS="fast32lo:a0.50 fast32h16lo:a0.50"
bash 05_distill/chain_frontends.sh
