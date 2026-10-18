# yam64h20-cliff
## Hypothesis
YAMNet's 64 bands / 25 ms window lose 0.04-0.05 between hop 16 ms (60 frames; yam64h16 a0.25_d8 0.711 at 4.57x,
a0.50_d8 0.727 at 3.77x) and hop 24 ms (40 frames; 0.658 at 6.69x, 0.689 at 5.59x), and hop 24 equals hop 32. So the
frame-rate cliff lies between 60 and 40 frames. Hop 20 ms (48 frames, window kept at 25 ms, overlapping like h16)
asks where. If the cost is roughly linear in frames, a0.25_d8 lands at ~0.685 at ~5.5x, above twofast32:a0.25_d8
(0.668 at ~6x) and level with twofast32:a0.50_d8 (0.701 / 0.694 at 5.33x) at a similar speed: a new frontier point
between 4.6x and 6x. Prediction: a0.25_d8 0.685 +- 0.015, a0.50_d8 0.705 +- 0.015.
Falsifier: a0.25_d8 <= 0.668 (h24's level, so the cliff is already complete at 48 frames).
Matched controls: yam64h16 and yam64h24 a0.25_d8 / a0.50_d8 (seed 1, rung C, WSD 56000, buzz+rain+human, _mmt rows).
## Changes
`05_distill/frontends.py`: new front end `yam64h20` (25 ms window, fft 512 as yamnet's, hop 320). `chain.sh`: cache C/V, main.py WSD 56000 for both students, mmt re-export.
## Results
## Conclusion
