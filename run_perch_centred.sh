#!/usr/bin/env bash
# perch_centred extraction, restarted in bounded bursts.
#
# Repeated tf.saved_model signature calls with the fixed bucketed batch size
# (embedders/perch_centred/embedder.py) still grow host RSS across many
# chunks -- slower than before the fix (which targeted per-call retrace
# accumulation), but not bounded. Rather than chase the remaining leak
# further, each burst runs under `timeout` and the whole extraction resumes
# safely across restarts: idents are fingerprinted, and an ident killed
# part-way carries extraction.incomplete and rebuilds (02_set/CLAUDE.md).
#
# Runs until 02_set/main.py itself exits 0 (nothing left to extract).
set -uo pipefail

WT=/home/luke/projects/buzzdetect-training/.local/worktrees/perch-centred
PY=/home/luke/projects/buzzdetect-training/.local/venv-perch-extract/bin/python
BURST_S=${BURST_S:-300}
SET=${SET:-medium}
CHUNK=${CHUNK:-8}

cd "$WT"
export PYTHONUNBUFFERED=1

i=0
while true; do
  i=$((i + 1))
  echo "== burst $i $(date -Is) =="
  BUZZDETECT_CHUNK_FRAMES=$CHUNK timeout "$BURST_S" "$PY" -u 02_set/main.py \
    --set "$SET" --embedder perch_centred --workers 1 --verbose
  rc=$?
  echo "== burst $i done rc=$rc $(date -Is) =="
  if [ "$rc" = 0 ]; then
    echo "== EXTRACTION COMPLETE =="
    break
  fi
  # rc=124 is timeout's own "killed for time", anything else (e.g. 137 for
  # an OOM SIGKILL) is also just resumed -- can_write()-style ident
  # fingerprinting handles both the same way.
done
