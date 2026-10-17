# d2-b64w16
## Hypothesis
`yam64h16` (64 bands, 25 ms window) scores 0.711/0.727 at 4.3x/3.8x; `fast32h16` (32 bands, 16 ms) 0.61-0.65 at 6.7-7.3x. They differ in both band count and window. A 64-band, 16 ms (fft 256) front end at hop 16 ms separates the two: if band count carries the gain it lands ~0.71 at 5-6x, above `twofast32:a0.50_d8` (0.70 at 5.33x). Falsifier: within 0.02 of the fast32h16 d8 twins (0.646/0.642 for a0.50_d8). Students a0.25_d8, a0.50_d8, rung C, WSD 56000.
## Changes
`05_distill/frontends.py`: new front end `b64w16h16`.

## Status: BLOCKED, not run (2026-10-17)
C-rung pack C__b64w16h16 built (48315/48395 slices; 80 decode failures tolerated). The V pack aborts:
46 of 6056 V slices have no b64w16h16 mel because their audio was renamed on the server disk around 2026-10-15
(e.g. 250722_1415.mp3 -> 260722_1415.mp3) while plan.csv keeps the old relpaths; V must be whole
(distill_train.py:142). One file, Wooster 2025-08-06/1_46/250806_1423.mp3, has no obvious counterpart
(250806_1523.mp3 present). Needs a plan remap or symlinks at the old names, and Luke's call on 1_46.
Resume: `tools/launch_job.sh distill.log -- bash chain.sh` once the audio paths resolve. Reported via loop_signal friction.

## Retry 2026-10-17 12:06 (batch 35 resume)
Still blocked: V mel fill fails on the same 46 slices (plan.csv relpaths `250722_1415.mp3` etc.; disk holds `260722_1415.mp3`). Needs plan remap or symlinks at old names; Luke's call on Wooster 1_46 `250806_1423.mp3`. Chain exits 1 at `[pack V]`. Not run.

## Fix 2026-10-17 (batch 36 fixer)
Cause: 13 Lily - Clover files were renamed on the server on ~2026-10-15 (25/16 -> 26 year prefix; two +1 h clock
fixes, Wooster 1_46 250806_1423 -> 250806_1523 and 250808_1605 -> 250808_1705; 1_148 260701_0314 -> 260630_1514).
All 13 match their durations.csv byte size, and `check_cache_align.py --renamed` reproduces the pre-rename
YAMNet mels to fp16 rounding (max|diff| ~2e-3, 12 of 12 files with a cached slice), so they are the planned audio
and 1_46 needed no call from Luke. Fix (main 5896368): `<distill_cache>/_shared/renamed.csv` maps old -> new
relpath, read by `cache.load_slice`; caches stay keyed by the planned relpath; cache_fe now stops up front on a
missing source file. chain.sh also refills rung C, so the 80 slices the first C fill dropped join the pack
(the pack rebuilds on its slice count, ~1 h), matching the other front ends' C packs.
