# d2-b64w16
## Hypothesis
`yam64h16` (64 bands, 25 ms window) scores 0.711/0.727 at 4.3x/3.8x; `fast32h16` (32 bands, 16 ms) 0.61-0.65 at 6.7-7.3x. They differ in both band count and window. A 64-band, 16 ms (fft 256) front end at hop 16 ms separates the two: if band count carries the gain it lands ~0.71 at 5-6x, above `twofast32:a0.50_d8` (0.70 at 5.33x). Falsifier: within 0.02 of the fast32h16 d8 twins (0.646/0.642 for a0.50_d8). Students a0.25_d8, a0.50_d8, rung C, WSD 56000.
## Changes
`05_distill/frontends.py`: new front end `b64w16h16`.
