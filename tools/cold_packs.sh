#!/usr/bin/env bash
# Move distillation shard packs off the root disk onto the server disk, leaving a symlink at the old path.
#
#   tools/launch_job.sh cold_packs.log -- bash tools/cold_packs.sh <teacher> <pack>...
#   e.g. ... -- bash tools/cold_packs.sh v4-ft-ps-e60-moderate C__yam64h16 B__lo32
#
# <pack> is a directory name under 05_distill/data/<teacher>/shards/. Each pack is copied (rsync), compared
# (rsync -c checksum dry run), and only then replaced by a symlink; a failed copy leaves the original alone.
# The server disk cannot store owner/group/permission bits, so rsync copies content and times only (-rlt).
# Safe to re-run: a pack that is already a symlink is skipped.
#
# The destination is a spinning disk. A symlinked pack still works, but a run that trains on it reads frames
# in random order from the HDD (~30 h for a 34 GB pack, against RAM of 23 GB): move it back to train on it.
#   rsync -a --remove-source-files <dest>/ <old path>/  (after `rm` of the symlink), or just re-run main.py
#   with the pack deleted so it rebuilds.
# Never move a pack a live job reads. shutil.rmtree (shards.py, stale-fingerprint rebuild) refuses a symlink:
# if a symlinked pack ever goes stale, remove the symlink by hand and rebuild.
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/.."
T=$1; shift
SRC=05_distill/data/$T/shards
DST="/media/server storage/distill-cache/_cold_packs/$T/shards"
mkdir -p "$DST"
for p in "$@"; do
  if [ -L "$SRC/$p" ]; then echo "[cold] $p: already a symlink, skip"; continue; fi
  [ -d "$SRC/$p" ] || { echo "[cold] $p: no such pack, skip"; continue; }
  echo "[cold] $p: $(du -sh "$SRC/$p" | cut -f1) copying"
  ionice -c3 rsync -rlt "$SRC/$p/" "$DST/$p/"
  if [ -n "$(rsync -rltnc --itemize-changes "$SRC/$p/" "$DST/$p/")" ]; then
    echo "[cold] $p: verify FAILED, original kept"; exit 1
  fi
  mv "$SRC/$p" "$SRC/$p.moving"
  ln -s "$DST/$p" "$SRC/$p"
  rm -rf "$SRC/$p.moving"
  echo "[cold] $p: moved ($(df -h --output=avail / | tail -1 | tr -d ' ') free on /)"
done
echo "[cold] done"
