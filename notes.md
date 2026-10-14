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

## Results

## Conclusion
