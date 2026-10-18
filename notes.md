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

## Results
Both students ran to WSD 56000 (the stop rule did not fire). mmt exports, 200 s GPU speed; mmt and stft headlines identical on all 8 checkpoints.

| student | headline | speed |
|---|---|---|
| b64w16h16 a0.25_d8 | 0.650 | 7.06x |
| b64w16h16 a0.50_d8 | 0.674 | 5.33x |
| fast32h16 a0.50_d8 s1 / s2 (falsifier twins) | 0.646 / 0.642 | 6.67x / 6.77x |
| twofast32 a0.50_d8 s1 / s2 (same speed) | 0.701 / 0.694 | 5.33x / 5.34x |
| yam64h16 a0.50_d8 | 0.727 | 3.77x |
| yam64h32 a0.25_d8 s2 (same speed as a0.25_d8) | 0.676 | 7.05x |

### vs fast32h16 a0.50_d8 s1 (same trunk; 32 -> 64 bands at 16 ms)
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.728 | 0.765 | +0.037 | 0.012 | 34 |
| 53 | 0.647 | 0.671 | +0.024 | 0.016 | 32 |
| 1_11 | 0.608 | 0.607 | -0.001 | 0.016 | 32 |
| 1_143 | 0.607 | 0.644 | +0.037 | 0.025 | 26 |
| 1_37 | 0.642 | 0.683 | +0.041 | 0.022 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.646 → this 0.674 (+0.028 ± 0.008)
- inclusive (sensitivity), same thresholds: 0.536 → 0.563 (+0.027)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.778 | 0.799 | +0.021 | 603 |
| untagged | 0.649 | 0.677 | +0.028 | 7485 |
| background | 0.642 | 0.676 | +0.034 | 9412 |
| quiet | 0.193 | 0.209 | +0.016 | 2231 |
| faint | 0.000 | 0.000 | +0.000 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

### vs twofast32 a0.50_d8 s1 (same speed)
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.807 | 0.765 | -0.042 | 0.016 | 34 |
| 53 | 0.674 | 0.671 | -0.003 | 0.021 | 32 |
| 1_11 | 0.628 | 0.607 | -0.021 | 0.026 | 32 |
| 1_143 | 0.673 | 0.644 | -0.029 | 0.032 | 26 |
| 1_37 | 0.725 | 0.683 | -0.042 | 0.021 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.701 → this 0.674 (-0.027 ± 0.011)
- inclusive (sensitivity), same thresholds: 0.590 → 0.563 (-0.027)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.810 | 0.799 | -0.011 | 603 |
| untagged | 0.705 | 0.677 | -0.028 | 7485 |
| background | 0.701 | 0.676 | -0.025 | 9412 |
| quiet | 0.233 | 0.209 | -0.024 | 2231 |
| faint | 0.000 | 0.000 | +0.000 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

### vs yam64h16 a0.50_d8 s1 (64 bands, 25 ms window)
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.824 | 0.765 | -0.059 | 0.020 | 34 |
| 53 | 0.657 | 0.671 | +0.014 | 0.017 | 32 |
| 1_11 | 0.678 | 0.607 | -0.071 | 0.022 | 32 |
| 1_143 | 0.715 | 0.644 | -0.071 | 0.024 | 26 |
| 1_37 | 0.760 | 0.683 | -0.077 | 0.020 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.727 → this 0.674 (-0.053 ± 0.009)
- inclusive (sensitivity), same thresholds: 0.613 → 0.563 (-0.050)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.815 | 0.799 | -0.016 | 603 |
| untagged | 0.731 | 0.677 | -0.054 | 7485 |
| background | 0.704 | 0.676 | -0.028 | 9412 |
| quiet | 0.255 | 0.209 | -0.046 | 2231 |
| faint | 0.000 | 0.000 | +0.000 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Reading: against its trunk twin, 64 bands gain +0.028 ± 0.008 on the headline, the folds up by about 1.5-2 SD apiece except 1_11 (-0.001 ± 0.016, flat); every tier moved up, quiet least (+0.016). The gain costs ~20% of the speed (6.67x -> 5.33x). At that speed twofast32 is ahead by 0.027 ± 0.011 (s1; 0.020 against s2), every fold down or flat (53 -0.003 ± 0.021 and 1_11 -0.021 ± 0.026 are unsure), so b64w16h16 a0.50_d8 is off the frontier. a0.25_d8 (0.650 at 7.06x) is also below yam64h32 a0.25_d8 at the same speed (0.676 s2). Against yam64h16 (same 64 bands, 25 ms window) the 16 ms window loses 0.053 ± 0.009, four folds down 3-4 SD (53 +0.014 ± 0.017, unsure), quiet tier -0.046: of yam64h16's ~0.08 lead over fast32h16, the window carries ~0.05 and the band count ~0.03.

## Conclusion
Prediction (~0.71 at 5-6x) failed: 0.674 at 5.33x. Falsifier (within 0.02 of the fast32h16 d8 twins) not met: +0.028 / +0.032. The 64 bands are a real gain at matched trunk, but most of yam64h16's advantage is its 25 ms window, and at the speed the extra bands cost, twofast32 does better. Neither student is on the frontier; no repeat needed (repeat rule unmet).
