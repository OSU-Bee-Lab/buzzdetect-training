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
Rung C, seed 1, WSD 56000, buzz+rain+human, `_mmtz` rows (re-export identical headlines to the unsuffixed rows).
Speed: same-session bench, 2 rounds, GPU rate / yamnet_large_general GPU rate, mean (`.local/bench/results_{1,2}.json`).

| student | headline | vs yam64h16 twin (paired) | x YAMNet GPU | twin x YAMNet |
|---|---|---|---|---|
| yam64h16lo a0.25_d8 | 0.679 | -0.032 +- 0.016 (0.711) | 6.97 | 5.23 |
| yam64h16lo a0.50_d8 | 0.677 | -0.050 +- 0.018 (0.727) | 5.22 | 4.21 |

Same-session neighbours: yam64h32 a0.25_d8_l5x0.50 s1 0.683 @ 7.18x (two-seed mean 0.670), yam64h32 a0.50_d8
0.687 @ 6.33x, twofast32 a0.50_d8 0.701 @ 5.74x.
Band cut buys 1.33x (a0.25) / 1.24x (a0.50) speed, short of the ~1.42x / ~1.38x predicted.
Per-fold (a0.25 / a0.50, eval-sampling SD only): 1_29 -0.103 +- 0.059 / -0.158 +- 0.073; 1_11 -0.052 +- 0.014 /
-0.028 +- 0.021; 1_37 -0.008 +- 0.047 / -0.053 +- 0.041; 53 and 1_143 within ~0.02 of zero. Weak per-fold evidence;
1_29 carries most of the mean drop in both widths.
Loudness tiers: background buzz falls -0.077 / -0.108 (9412 frames, the largest tier), loud rises +0.035-0.038
(603 frames): the lost >2.5 kHz band carries the cue for buzz under other sound, not for loud buzz.
WSD: a0.25 still gaining slowly (hit@K +1.87, +0.93 over the last two doublings, stop rule met); a0.50 peaked at
7000 steps (0.690) and drifted down to 0.677 while hit@K crept up (+0.55, +0.36), so its budget is past useful.
## Conclusion
Falsified: a0.25_d8 0.679 < 0.683 (the speed half held at 6.97x > 6.5x). The 125-2500 Hz cut costs 0.03-0.05
headline, mostly in background-tier buzz, unlike the rung-B fast32lo twins that tied. At ~7x it ties
yam64h32 a0.25_d8_l5x0.50 (0.683 s1 / 0.670 two-seed @ 7.18x) without beating it; a0.50_d8 (0.677 @ 5.22x) is
dominated by twofast32 a0.50_d8 (0.701 @ 5.74x). Neither moves the frontier, and neither triggers the repeat rule.
Takeaway: the >2.5 kHz band is worth keeping on the 64-band trunk; cheaper front ends should come from fewer DFT bins
elsewhere (coarser bands above 2.5 kHz) rather than dropping it.
