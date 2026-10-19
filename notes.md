# yam64h16sp2
## Hypothesis
The `mmtz` front end is FLOP-bound in the windowed-DFT MatMul (frames/s x window x 2 x DFT bins, mmt-overlap), and
yam64h16lo showed the cheap way to cut bins (dropping >2.5 kHz) loses background-tier buzz (-0.03/-0.05). Keep
the band, cut the bins instead: above ~1 kHz yam64h16's mel bands are >= 62.5 Hz apart while the 512-point DFT
samples every 31.25 Hz, finer than the 25 ms window resolves (mainlobe ~80 Hz). `yam64h16sp2` reads every
2nd bin there (each kept bin weighted by the gap it stands for, so band sums approximate the full grid's:
log-mel corr 0.986 with yam64h16 on the fixture), 132 DFT bins of 235. Same window, hop, bands, trunk.
Prediction (mmtz, same-session timing): a0.25_d8 0.705 +- 0.02 at ~6.2x (twin 0.711 at ~5.25x), a0.50_d8
0.72 +- 0.02 at ~4.9x (twin 0.727 at ~4.2x). At ~6.2x the best is yam64h32 a0.50_d8 0.687 @ 6.06x, so
a0.25_d8 would be a frontier step.
Falsifier: a0.25_d8 < 0.690 (no better than what sits at ~6x), or a0.25_d8 under 1.1x its twin's speed in the
same session.
Matched controls: yam64h16 a0.25_d8 / a0.50_d8 (seed 1, rung C, WSD 56000, buzz+rain+human), `_mmtz` rows.
