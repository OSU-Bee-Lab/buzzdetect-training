# yam64h32
## Hypothesis
Distillation arm, the lead `yam64h16` left. On YAMNet's 64 bands the 16 ms hop cost almost nothing against
the 10 ms twins (-0.002 / -0.008 +- 0.007), while frame rate is the largest speed term. `yam64h32` halves
the frame rate again: 64 bands, 125-7500 Hz, hop 32 ms (30 frames per patch), the window grown to the hop
(32 ms, fft 512) so no audio falls between frames (as `fast32h32` did).

Prediction: `yam64h32:a0.25_d8` / `a0.50_d8` keep within ~0.01-0.02 of their `yam64h16` twins (0.711 / 0.727)
and time well above them (front end alone should approach fast32h32's 8.4x territory, minus the 64-band
mel), so they land above the current line between `yam64h16:a0.25_d8` (0.711 at 4.57x) and
`twofast32:a0.25_d8` / `fast32h16` (0.668 at ~6x, 0.646 at 6.7x).
Falsifier: the hop-32 cost on 64 bands is ~0.03 or more against the hop-16 twins (frame rate stops being
free below ~30 frames per patch), or the bench puts them no faster than `yam64h16`.
Matched comparators (same teacher `v4-ft-ps-e60-moderate`, rung C, WSD 7000-56000, seed 1, buzz+rain+human,
same trunks, mmt): `fe_C_yam64h16_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt`, `..._a0.50_d8_...`.

## Changes
- `05_distill/frontends.py`: `yam64h32` registered. The rest of the `05_distill/` diff is `exp/yam64h16`'s
  code as a patch (the `mmt` front end, `reexport.py`, bench), not in main.
- `bench.sh`: random-weight speed (pad-free `mmt`, 200 s, GPU), run as the chain's first step.
- `chain.sh`: bench, then `main.py --rung C --seed 1 --wsd-max 56000 --wsd-stop 1.3` on `yam64h32:a0.25_d8`
  and `a0.50_d8` (buzz+rain+human), then `reexport.py --fe-impl mmt --suffix _mmt` on the judged branches.
- Smoke: `a0.25_d8@yam64h32+mmt` builds and exports on CPU (40k params, 0.82 MMACs).

### Speed bench before training (random weights, pad-free `mmt`, 200 s, GPU; `bench.log`)
| graph | x YAMNet | MMACs |
|---|---|---|
| front end alone, yam64h32 | 6.62 | |
| **yam64h32 a0.25_d8** | **6.70** | 0.8 |
| **yam64h32 a0.50_d8** | **6.00** | 2.9 |
| front end alone, yam64h16 | 5.37 | |
| yam64h16 a0.25_d8 | 4.20 | 1.6 |
| yam64h16 a0.50_d8 | 3.79 | 5.8 |
| twofast32 a0.50_d8 | 5.33 | 3.0 |
| twofast32 a0.25_d8 | 6.12 | 0.8 |
| front end alone, fast32h16 | 8.90 | |
| fast32h16 a0.25 | 6.80 | 1.6 |

Reading before training: halving the frame rate makes `yam64h32:a0.50_d8` faster than `twofast32:a0.50_d8`
(6.00x vs 5.33x) and `a0.25_d8` level with `fast32h16:a0.25` (6.70x vs 6.80x), the 64-band front end
(6.6x alone) now the ceiling. (`a0.25_d8` timing above its own front end is session noise, ~+-0.3x.)
So `a0.50_d8` is above the line if it holds ~0.70 (`twofast32:a0.50_d8` 0.694-0.701 is slower), and
`a0.25_d8` if it beats ~0.67 (`twofast32:a0.25_d8` 0.668 at 6.1x, `fast32h16:a0.50_d8` 0.646 at 6.7x).

## Results

## Conclusion
