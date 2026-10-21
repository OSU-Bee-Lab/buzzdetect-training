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
| USB NVMe | `/dev/sdc` | not mounted | Sabrent 1 TB, **holds Luke's data** | n/a |

### The NVMe is not blank

Luke says macOS reads data on it. Linux shows no partition table and no filesystem
signature it recognises (likely APFS or HFS+). **Do not partition, format or
`wipefs` it.** Ask Luke what it is. If it must be read on Linux, mount it
read-only (`apfs-fuse` for APFS, `hfsprogs` or the kernel `hfsplus` driver for
HFS+). It becomes a migration target only if Luke empties it or says to wipe it.

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
possibly lower from the NTFS-FUSE CPU cost on the i7-2600. The USB NVMe is untested.

## What is on the SSD

| Path | Size | Verdict |
|---|---|---|
| `05_distill/data/v4-ft-ps-e60-moderate/shards` | 313 GB | **Move** |
| `05_distill/data/v4-ft-ps-e60-moderate-ens3/shards` | 75 GB | **Move** |
| `02_set/sets/medium` | 90 GB (70 GB embeddings, 20 GB audio) | Keep for now |
| `02_set/sets/moderate` | 57 GB | Move; final-confirmation set only |
| `02_set/sets/lite` | 8 GB | Keep |
| `.local/worktrees` | 19 GB | Prune finished ones, do not move |
| `.local/venv-onnx`, `venv-quant*`, `int8-ptq` | ~9 GB | Move (closed distillation arm) |
| `05_distill/data/*/{eval,models,runs}`, `_contaminated_*` | ~13 GB | Leave: small, read by ladder tooling |
| `~/anaconda3` | 36 GB | Keep |
| `03_train`, `models` | ~1 GB | Keep |

Moving both `shards` directories (388 GB) alone brings the SSD to roughly 55% used.

Already on the HDD: `distill_cache` (511 GB: `_mel` 275 GB, `_cold_packs` 159 GB),
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

- The set's embeddings are only read once, sequentially. Moving them to the HDD
  costs a load of ~20 s (depth8, 4 GB) to ~2 min (depth4, 16 GB) per run. A CV run
  is tens of minutes to hours, so that is noise. After the first read the file
  sits in the page cache (23 GB RAM).
- **Keep `.local/stream_scratch` on SSD or NVMe.** On the HDD the random reads are
  ~100 IOPS against 7,000, and training would slow by an unknown, probably large,
  factor. Do not symlink `.local` wholesale. Symlink individual directories.
- Scripts that set `TRUNK_STREAM=1`: `tools/queue_v4_moderate.sh`,
  `tools/queue_v4_moderate_ship.sh`, and in worktrees `ens3-moderate/chain.sh`,
  `d1-ens-teacher/chain_ship.sh`, `ps-depth4/run_depth4.sh`,
  `llrd-depth4/chain_llrd.sh`, `pauc-l03/chain_b38.sh` (the llrd-depth4 run). The
  depth8 chains may or may not set it through a `D8` variable in `chain_b39.sh`;
  this was not checked. `03_train/log.jsonl` does not record env vars.

## Plan

1. **Preconditions**: no chain running; Luke has said where the NVMe stands.
2. **Compare the duplicates**: `rsync -rn --itemize-changes` of the two `shards`
   dirs against `distill-cache/`. Decide what is already migrated.
3. **Move the `shards` dirs** to the HDD. `05_distill/data` is configured through
   `config.py` / `paths.local.json`; `shards` is under `05_distill/data`, not under
   `distill_cache`. Add a `paths.local.json` key, or symlink the two directories.
   Check how `05_distill` resolves them before choosing.
4. **Test a symlink on the HDD first.** NTFS-3G symlink support is weaker than
   ext4 and the mount has no Unix permissions. Create one, resolve it from Python,
   and read through it.
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

- What is on the NVMe, and can it be reformatted?
- Which parts of `05_distill` can leave the machine entirely?
- Is the HDD NTFS by choice? An ext4 partition would remove the FUSE limits, but it
  needs the drive emptied or repartitioned and is Luke's call.
