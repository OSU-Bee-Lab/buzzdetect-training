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
