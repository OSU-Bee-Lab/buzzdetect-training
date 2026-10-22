# Disk migration guide

Written 2026-10-20 for a future agent moving data off the root SSD between eras.
Everything here was measured or read on that date; re-check sizes before acting.
Nothing in this guide has been executed yet.

## Why

Root SSD `/dev/sda2` (SanDisk SATA, 937 GB) was 97% full: 855 GB used, 35 GB free.
Training needs room for `.local/stream_scratch` (a depth4 pool is ~16 GB of
float16) and for extraction output, so the margin is too thin.

## Drives

| Drive | Device | Mount | Type | Free (2026-10-20) |
|---|---|---|---|---|
| Root SSD | `/dev/sda2` | `/` | ext4, SATA | 35 GB |
| HDD | `/dev/sdb1` | `/media/server storage` | **NTFS via FUSE (`fuseblk`)**, WD 18 TB, 7200-class | 12 TB |

### Rules for the HDD (Luke, 2026-10-21)

- **It stays NTFS.** It holds critical data: never repartition, reformat or
  convert it.
- **Nothing is deleted without Luke's approval.** Anything may *move* to the
  HDD; leaving the machine, or deleting, is his call every time.

## Measured speeds

Run with `dd` (direct IO) and a 4 KB random-read script, while the drives were idle
and batch 39 was running.

| | SSD | HDD |
|---|---|---|
| Sequential write, 3 GB | 327 MB/s | 118 MB/s |
| Sequential read, 3 GB | 424 MB/s | **not measured** (1.8 GB/s was RAM cache) |
| Random 4 KB read | 7,264 IOPS | **not measured** (49,771 was RAM cache) |
| 2,000 x 20 KB files, read back | 0.4 s (cached) | 17 s |

The HDD's direct-IO reads are fake because FUSE ignores `O_DIRECT`. To get a true
cold read, read a file that was never cached (or `drop_caches` as root, which needs
Luke) and time it. Expect 100-200 MB/s sequential and ~100 IOPS random, and
possibly lower from the NTFS-FUSE CPU cost on the i7-2600.

**Measured 2026-10-21: cold sequential reads deliver 30-48 MB/s** (three 1-2 GB
reads of an uncached 12 GB file under `distill-cache/_cold_packs`), well below
the 100-200 MB/s guessed above. The disk itself read ~140 MB/s of sectors during
the 44 MB/s read, so the NTFS-FUSE layer, not the platter, is the limit.

**The kernel driver removes most of that (2026-10-21).** Same file, an unread
2 GB region, mounted read-only with the in-kernel `ntfs3` driver instead of the
FUSE `ntfs-3g`: **158 MB/s**, 3-5x the FUSE reads. One measurement. The drive
mounts at boot from `/etc/fstab` with type `auto`, which picks FUSE; switching
means changing that to `ntfs3`. **Switched 2026-10-21** (Luke; the data is backed
up). A read-write trial passed first: a 1 GB file written under `ntfs3` (118
MB/s, same as FUSE: writes are disk-bound) hashed identically after a remount
and through FUSE. `/etc/fstab`'s line is now type `ntfs3` with
`uid=1001,gid=1003,umask=000`; the old file is at `/etc/fstab.bak-2026-10-21`.
Gid 1003 is `hdd-share` (members `luke`, `beelab`), so other lab users can write.
Under `ntfs3`, entries that carry their own stored owner/group/mode keep them
over the mount options (`noacsrules`, which overrode that, is not in this
kernel's driver): 1,195 entries in `experiments/` stored `luke:luke`, and were
re-grouped to `hdd-share` (`chgrp`, metadata only). 122 raw recordings there are
`644`: other users can read, rename or delete them but not edit them in place. The pipeline's `audio_root`
and `distill_cache` read and write normally. To go back: restore that backup,
`sudo systemctl daemon-reload`, then unmount and remount the drive.

## What is on the SSD

| Path | Size | Verdict |
|---|---|---|
| `05_distill/data/v4-ft-ps-e60-moderate/shards` | 313 GB | **Moved 2026-10-22** (symlinked packs) |
| `05_distill/data/v4-ft-ps-e60-moderate-ens3/shards` | 75 GB | **Moved 2026-10-22** (symlinked packs) |
| `02_set/sets/medium` | 90 GB (70 GB embeddings, 20 GB audio) | Keep for now |
| `02_set/sets/moderate` | 57 GB | Move; final-confirmation set only |
| `02_set/sets/lite` | 8 GB | Keep |
| `.local/worktrees` | 19 GB | Prune finished ones, do not move |
| `.local/venv-onnx`, `venv-quant*`, `int8-ptq` | ~9 GB | Move (closed distillation arm) |
| `05_distill/data/*/{eval,models,runs}`, `_contaminated_*` | ~13 GB | Leave: small, read by ladder tooling |
| `~/anaconda3` | 36 GB | Keep |
| `03_train`, `models` | ~1 GB | Keep |

**Done 2026-10-22:** every pack under both `shards` directories (388 GB, 30 packs) went to
`distill-cache/_cold_packs/<teacher>/shards/` with `tools/cold_packs.sh` (copy, checksum
verify, replace by a symlink; ~1 h at the HDD's write speed). The `shards` directories
stay on the SSD and hold only symlinks. The SSD went from 99% used (9.5 GB free) to 56% (397 GB free).

Already on the HDD before that: `distill_cache` (511 GB: `_mel` 275 GB, `_cold_packs` 159 GB),
set by `"distill_cache"` in `paths.local.json`. The HDD **also** holds
`v4-ft-ps-e60-moderate` (64 GB) and `-ens3` (14 GB) under `distill-cache/` with the
same names as the SSD dirs. That looks like a partial earlier migration. Run
`rsync -rn` and compare before copying, or you will double-copy or overwrite.

Not surveyed: `~/projects/buzzdetect` (119 GB) and `~/projects/buzzdetect-agentic`
(16 GB), outside this repo.

## Before you touch anything

1. `pgrep -af "[c]hain"`. If a chain runs (batch 39's `chain2_b39.sh` and
   `mixstyle-xfold/chain_b39.sh` were alive on 2026-10-20), do not move, rename or
   edit anything it reads: stage scripts, cache dirs, `paths.local.json`. Wait.
2. Symlink first, rename after: copy the data, point a symlink at the copy, check
   the pipeline still reads it, then remove the original. Never `mv` out from under
   a reader.
3. Free space on the target: `df -h`.
4. Do not train or run CV on `moderate` as an experiment. Moving it is fine.

## Streaming training pools (the main hazard)

`TRUNK_STREAM=1` (`03_train/train.py`, `_StreamPool`) reads a fold's embeddings
sequentially from the set, writes them to one float16 file under
`cfg.DIR_STREAM_SCRATCH` (`.local/stream_scratch`, `config.py`), memory-maps it for
**random access**, and deletes it after the fold.

- The set's embeddings are read sequentially. Under FUSE, at the measured 30-48 MB/s, a
  medium depth8 cache (4 GB, 1,434 files) costs **~1.5-2.5 min per CV** from the
  HDD, about 1% of a ~3 h CV, because it fits in the page cache (23 GB RAM, ~14 GB
  free) and is read once. A pool that doesn't fit is re-read every fold: a depth4
  pool (16 GB) or a moderate two-view pool (~18 GB) costs ~6-10 min per fold,
  tens of minutes per CV. So: medium's active embedders can live on the HDD at
  little cost; the biggest pools are where it shows. Under `ntfs3` (158 MB/s, the driver now in use)
  the depth8 load drops to ~30 s per CV and the big pools to ~2 min per fold.
- **Keep `.local/stream_scratch` on the SSD.** On the HDD the random reads are
  ~100 IOPS against 7,000, and training would slow by an unknown, probably large,
  factor. Do not symlink `.local` wholesale. Symlink individual directories.
- Scripts that set `TRUNK_STREAM=1`: `tools/queue_v4_moderate.sh`,
  `tools/queue_v4_moderate_ship.sh`, and in worktrees `ens3-moderate/chain.sh`,
  `d1-ens-teacher/chain_ship.sh`, `ps-depth4/run_depth4.sh`,
  `llrd-depth4/chain_llrd.sh`, `pauc-l03/chain_b38.sh` (the llrd-depth4 run). The
  depth8 chains may or may not set it through a `D8` variable in `chain_b39.sh`;
  this was not checked. `03_train/log.jsonl` does not record env vars.

## Plan

1. **Preconditions**: no chain running.
2-4. **Done 2026-10-22: the `shards` packs are moved** (see "What is on the SSD").
   Each pack is a symlink on the SSD pointing to its copy on the HDD, the same layout
   as the 2026-10-15 cold packs. Nothing reads them while distillation is closed.
5. **Move the closed-arm venvs and `int8-ptq`.** Check `paths.local.json`'s
   `"python"` and `tools/python_path.sh` first: if any tool resolves to a `.local`
   venv, leave it.
6. **Prune finished worktrees** (do not move). Candidates on 2026-10-20:
   `pseudo-label` (3.8 GB), `ps-sharedtiled`, `ps-untiled`, `ps-fast`,
   `yamnet-aves-head`. Confirm each branch is merged or archived first
   (`archive/README.md` covers `refs/archive/<slug>`).
7. **Move `moderate`** (57 GB) with a symlink, once nothing reads it.
8. **Measure**: time a cold read of one `medium` embedder from the HDD and run a
   short training job with the set symlinked. Record the real load time here.
9. **Optional, `medium`**: only after step 8, and only if the numbers hold.

## Verify after each move

- `df -h /` shows the freed space.
- A tiny stage-3 smoke run reads through the symlink (`LOOP.md`: the smoke set must
  contain the embedder, or the chain dies).
- `git status` is clean: the moved paths are gitignored, but check that no tracked
  file turned into a symlink.

## Open questions for Luke

None open as of 2026-10-21 (answers are in the HDD rules above).
