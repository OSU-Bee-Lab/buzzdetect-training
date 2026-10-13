# yam64h16
## Hypothesis
Distillation arm (IDEAS D2). After `matmul-stft` and `twofast32-d8-s2`, the frontier jumps from the YAMNet
front end (`yamnet:a0.25_d8` 0.713 at 3.24x) to `twofast32:a0.50_d8` (0.694-0.701 at 5.33x) with nothing
between. Every fast front end changed three things at once against YAMNet's: the frame rate (hop 10 ->
16 ms), the band count (64 -> 32) and the window (25 -> 16 ms, or two windows). `yam64h16` changes only
the frame rate: YAMNet's 25 ms window, 64 bands, 125-7500 Hz, hop 256 (60 frames per patch).

Prediction: the 64 bands and 25 ms window keep most of what the fast front ends lose, so
`yam64h16:a0.25_d8` / `a0.50_d8` land above the line joining 0.713 at 3.24x and 0.70 at 5.33x by more
than 0.02, at about 4-5x.
Falsifier: they land on that line or under it (frame rate, not bands or window, is what the fast front
ends lose to), or the random-weight bench puts them no faster than the YAMNet-front-end students.
Matched comparators (same teacher, rung C, WSD 7000-56000, seed 1, buzz+rain+human, same trunk):
`yamnet:a0.25_d8` (0.713), `yamnet:a0.50_d8` (0.735) for the frame-rate cost; `twofast32:a0.50_d8` /
`a0.25_d8` for the frontier.

## Changes
- `05_distill/frontends.py`: `yam64h16` registered. The rest of the `05_distill/` diff is
  `exp/twofast32-d8`'s code as a patch (the `mmt` front end, `reexport.py`), not in main.
- `bench.sh`: random-weight speed (pad-free `mmt`, 200 s, GPU) before any training.
- `chain.sh`: `main.py --rung C --seed 1 --wsd-max 56000 --wsd-stop 1.3` on `yam64h16:a0.25_d8` and
  `a0.50_d8` (buzz+rain+human), then `reexport.py --fe-impl mmt --suffix _mmt` on the judged branches.

### Speed bench before training (random weights, pad-free `mmt`, 200 s, GPU; `bench.log`)
| graph | x YAMNet | MMACs |
|---|---|---|
| front end alone, yam64h16 | 4.99 | |
| **yam64h16 a0.25_d8** | **4.66** | 1.6 |
| yam64h16 a0.375_d8 | 4.19 | 3.4 |
| **yam64h16 a0.50_d8** | **3.74** | 5.8 |
| yam64h16 a0.25 | 4.24 | 3.1 |
| front end alone, yamnet | 3.60 | |
| yamnet a0.25_d8 | 3.13 | 2.5 |
| yamnet a0.50_d8 | 2.63 | 8.9 |
| twofast32 a0.50_d8 | 5.02 | 3.0 |
| twofast32 a0.25_d8 | 6.23 | 0.8 |

(`twofast32:a0.50_d8` timed 5.02x in this session against 5.33-5.36x on 2026-10-11/12: a 6% session
spread on one graph, so read these to about +-0.3x.)

Reading before training: the front end alone is 4.99x, under `twofast32:a0.50_d8`'s whole graph
(5.0-5.3x, 0.694-0.701). So every `yam64h16` student is slower than that student, and to be above the
line it must beat ~0.70 by more than seed noise: `a0.25_d8` at 4.66x would have to match or beat its own
10 ms-hop twin (0.713), `a0.50_d8` at 3.74x has to clear 0.713 (yamnet a0.25_d8 is slower, 3.13x). The
prediction is sharpened accordingly: if the hop costs what it cost fast32 (~0.035), the pair lands near
0.68 / 0.70, on or under the line, and the idea is falsified on speed + headline together. Run anyway:
it is the only measurement that separates frame rate from bands/window, and a hop cost well under 0.035
on 64 bands would put `a0.50_d8` on the frontier.

## Results
Headline at WSD 56000 (neither trunk was stopped early); speed is the `_mmt` row's x YAMNet at 200 s on the GPU.

| student | headline / speed | 10 ms-hop twin (`yamnet`) | hop cost | best student at its speed or faster | over it |
|---|---|---|---|---|---|
| `yam64h16:a0.25_d8` | 0.711 / 4.57x | 0.713 / 3.24x | -0.002 ± 0.007 | `twofast32:a0.50_d8` 0.701 (s1), 0.694 (s2) / 5.33x | +0.010 ± 0.009 (s1), +0.017 (s2) |
| `yam64h16:a0.50_d8` | 0.727 / 3.77x | 0.735 / 2.60x | -0.008 ± 0.007 | `yam64h16:a0.25_d8` 0.711 / 4.57x | +0.016; +0.026 ± 0.009 over `twofast32:a0.50_d8` s1 |

(± SD is eval sampling only; headline seed noise on a student is 0.01-0.02.)

Step-budget curves (headline at 7000 / 14000 / 28000 / 56000):
- `a0.25_d8`: 0.684 / 0.695 / 0.706 / 0.711; hit@K +3.22, +0.98, +0.94. `yamnet` twin: 0.676 / 0.691 / 0.705 / 0.713.
- `a0.50_d8`: 0.708 / 0.717 / 0.725 / 0.727; hit@K +1.87, +1.13, +0.27. `yamnet` twin: 0.727 / 0.731 / 0.735 / 0.735.

Measured speeds match the random-weight bench (4.66x / 3.74x). `tools/results.py` tables (results.log):

### a0.25_d8: `yamnet` (10 ms hop) -> `yam64h16` (16 ms hop)

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.799 | 0.801 | +0.002 | 0.008 | 34 |
| 53 | 0.651 | 0.663 | +0.012 | 0.015 | 32 |
| 1_11 | 0.676 | 0.688 | +0.012 | 0.014 | 32 |
| 1_143 | 0.690 | 0.680 | -0.010 | 0.017 | 26 |
| 1_37 | 0.749 | 0.723 | -0.026 | 0.023 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.713 → this 0.711 (-0.002 ± 0.007)
- inclusive (sensitivity), same thresholds: 0.600 → 0.594 (-0.006)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.811 | 0.817 | +0.006 | 603 |
| untagged | 0.722 | 0.717 | -0.005 | 7485 |
| background | 0.682 | 0.692 | +0.010 | 9412 |
| quiet | 0.249 | 0.231 | -0.018 | 2231 |
| faint | 0.067 | 0.000 | -0.067 | 35 |

2026-10-13 07:14:53.652107: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:479] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2026-10-13 07:14:53.677472: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:10575] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2026-10-13 07:14:53.677516: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1442] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2026-10-13 07:14:54.534069: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT

### a0.50_d8: `yamnet` (10 ms hop) -> `yam64h16` (16 ms hop)

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.829 | 0.824 | -0.005 | 0.009 | 34 |
| 53 | 0.654 | 0.657 | +0.003 | 0.013 | 32 |
| 1_11 | 0.713 | 0.678 | -0.035 | 0.018 | 32 |
| 1_143 | 0.717 | 0.715 | -0.002 | 0.016 | 26 |
| 1_37 | 0.764 | 0.760 | -0.004 | 0.016 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.735 → this 0.727 (-0.008 ± 0.007)
- inclusive (sensitivity), same thresholds: 0.618 → 0.613 (-0.005)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.828 | 0.815 | -0.013 | 603 |
| untagged | 0.742 | 0.731 | -0.011 | 7485 |
| background | 0.704 | 0.704 | +0.000 | 9412 |
| quiet | 0.257 | 0.255 | -0.002 | 2231 |
| faint | 0.067 | 0.000 | -0.067 | 35 |

2026-10-13 07:16:39.188989: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:479] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2026-10-13 07:16:39.212898: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:10575] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2026-10-13 07:16:39.212948: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1442] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2026-10-13 07:16:40.076595: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT

### `twofast32:a0.50_d8` s1 (5.33x) -> `yam64h16:a0.25_d8` (4.57x)

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.807 | 0.801 | -0.006 | 0.014 | 34 |
| 53 | 0.674 | 0.663 | -0.011 | 0.013 | 32 |
| 1_11 | 0.628 | 0.688 | +0.060 | 0.028 | 32 |
| 1_143 | 0.673 | 0.680 | +0.007 | 0.024 | 26 |
| 1_37 | 0.725 | 0.723 | -0.002 | 0.022 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.701 → this 0.711 (+0.010 ± 0.009)
- inclusive (sensitivity), same thresholds: 0.590 → 0.594 (+0.004)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.810 | 0.817 | +0.007 | 603 |
| untagged | 0.705 | 0.717 | +0.012 | 7485 |
| background | 0.701 | 0.692 | -0.009 | 9412 |
| quiet | 0.233 | 0.231 | -0.002 | 2231 |
| faint | 0.000 | 0.000 | +0.000 | 35 |

2026-10-13 07:18:24.859961: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:479] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2026-10-13 07:18:24.884183: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:10575] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2026-10-13 07:18:24.884238: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1442] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2026-10-13 07:18:25.736300: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT

### `twofast32:a0.50_d8` s1 (5.33x) -> `yam64h16:a0.50_d8` (3.77x)

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.807 | 0.824 | +0.017 | 0.016 | 34 |
| 53 | 0.674 | 0.657 | -0.017 | 0.017 | 32 |
| 1_11 | 0.628 | 0.678 | +0.050 | 0.024 | 32 |
| 1_143 | 0.673 | 0.715 | +0.042 | 0.022 | 26 |
| 1_37 | 0.725 | 0.760 | +0.035 | 0.015 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.701 → this 0.727 (+0.026 ± 0.009)
- inclusive (sensitivity), same thresholds: 0.590 → 0.613 (+0.023)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.810 | 0.815 | +0.005 | 603 |
| untagged | 0.705 | 0.731 | +0.026 | 7485 |
| background | 0.701 | 0.704 | +0.003 | 9412 |
| quiet | 0.233 | 0.255 | +0.022 | 2231 |
| faint | 0.000 | 0.000 | +0.000 | 35 |

Interpretation.
- **Hop cost on 64 bands is within noise**: -0.002 ± 0.007 (`a0.25_d8`) and -0.008 ± 0.007 (`a0.50_d8`), against
  ~0.035 on `fast32` -> `fast32h16`. The step-budget curves track the twins' at every budget for `a0.25_d8`; `a0.50_d8`
  sits 0.008-0.019 under its twin throughout. So with YAMNet's 64 bands and 25 ms window, hop 10 -> 16 ms is nearly
  free; what the fast front ends lose is mostly bands / window, not frame rate. The falsifier ("lands on the line")
  did not happen, but the prediction's margin (> 0.02 above the line) was not met either.
- Folds, hop cost: `a0.25_d8` moves no fold beyond its SD (`1_37` -0.026 ± 0.023, unsure). `a0.50_d8` loses on
  `1_11` (-0.035 ± 0.018, about 2 SD of eval sampling, one seed); the rest are flat.
- Tiers, hop cost: `a0.25_d8` flat (loud +0.006, untagged -0.005, background +0.010); `a0.50_d8` loud -0.013,
  untagged -0.011, background 0.000: small and the same sign, consistent with a hop cost of about 0.01.
- **Frontier**: both students are on the non-dominated line. `a0.25_d8` (0.711 at 4.57x) matches `yamnet:a0.25_d8`
  (0.713) 1.4x faster and is +0.010 ± 0.009 / +0.017 over the two `twofast32:a0.50_d8` seeds (weak: inside seed noise).
  `a0.50_d8` (0.727 at 3.77x) is above every faster student by +0.016 and dominates `yamnet:a0.25_d8`, `a0.25` and
  `a0.375_d8` (0.707-0.716 at 2.9-3.2x). Against `twofast32:a0.50_d8` its gain is in untagged (+0.026) and quiet
  (+0.022) with `1_11`, `1_143`, `1_37` up 0.035-0.050 (1.9-2.3 SD) and `53` -0.017 ± 0.017.
- Neither margin over everything at its speed or faster exceeds 0.02, so the repeat rule does not fire
  (`ladder_record.py frontier` lists neither).

## Conclusion
Weak-to-moderate gain; the structural finding is the stronger part. On YAMNet's 64 bands and 25 ms window the
16 ms hop costs -0.002 / -0.008 (± 0.007) against ~0.035 on 32 bands, so the fast front ends' loss is bands / window.
`yam64h16:a0.50_d8` (0.727 at 3.77x) and `a0.25_d8` (0.711 at 4.57x) fill the hole between `yamnet` and `twofast32`
and replace the `yamnet` a0.25-a0.375 students on the frontier (same headline, 1.2-1.4x faster), but each is only
+0.010 to +0.017 above the best faster student, within seed noise, single seed. Tiers: against the 10 ms twins
nothing moves beyond 0.013; against `twofast32` the gain is untagged and quiet, background flat. Leads: a 64-band
front end at a longer hop still (hop 20-32 ms), or 64 bands with `twofast32`'s cheaper window, since frame rate is
the largest speed term and now looks nearly free in headline.
