# yam64h16lo
## Hypothesis
mmt-overlap found the front end is ~75-80% of a 25 ms-window student's GPU time even with `mmtz`, FLOP-bound in
the windowed-DFT MatMul (frames/s x window x 2 x DFT bins). yam64h16's 64 bands over 125-7500 Hz need 235 DFT
bins; over 125-2500 Hz they need 75 (no empty bands, 15 one-bin bands), so that MatMul and the elementwise
magnitude ops behind it shrink ~3x. The `lo` (100-2500 Hz) twins of fast32 and fast32h16 tied their full-band
twins (FRONTENDS.md: 0.597 vs 0.606, 0.564 vs 0.562, rung B) and buzz frames differ from negatives nearly flat
across frequency, so the band cut should cost little.
Prediction (mmtz, same-session timing): yam64h16lo a0.25_d8 0.70 +- 0.02 at ~7.5x (yam64h16 a0.25_d8 0.711 at 5.27x
mmtz), a0.50_d8 0.715 +- 0.02 at ~5.8x: a frontier step at ~7x, where the best is ~0.683 (yam64h32
a0.25_d8_l5x0.50 s1, two-seed mean 0.670) at 7.1x.
Falsifier: a0.25_d8 < 0.683, or a0.25_d8 slower than 6.5x in the same session as its twin.
Matched controls: yam64h16 a0.25_d8 / a0.50_d8 (seed 1, rung C, WSD 56000, buzz+rain+human), `_mmtz` rows.
## Changes
`05_distill/frontends.py`: front end `yam64h16lo` (25 ms window, hop 256, 64 bands 125-2500 Hz); `mmtz` impl and
the `yam64h20` front end brought over from exp/mmt-overlap. `chain.sh`: cache C/V, main.py WSD 56000 for both
students, mmtz re-export, then a same-session timing (2 rounds) of both students, their yam64h16 twins and the
~6-7x frontier students into `.local/bench/results_{1,2}.json`.
## Results
## Conclusion
