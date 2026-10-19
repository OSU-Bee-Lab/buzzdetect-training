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

## Results
Same-session GPU timing (bench_arch time, 200 s, 15 repeats, two passes agree to <0.5%; YAMNet 3657 s/s):

| student (WSD 56000, _mmtz) | GPU s/s | x YAMNet | headline | twin (yam64h16) | twin x YAMNet | twin headline |
|---|---|---|---|---|---|---|
| sp2 a0.25_d8 | 22043 | 6.03 | 0.699 | 19214 | 5.25 | 0.711 |
| sp2 a0.50_d8 | 17062 | 4.67 | 0.718 | 15368 | 4.20 | 0.727 |

Same session, for reference: yam64h32 a0.50_d8 6.34x (0.687), yam64h32 a0.25_d8_l5x0.50 7.17x (0.683), twofast32 a0.50_d8 5.73x (0.701).

### a0.25_d8 vs yam64h16 a0.25_d8 (tools/results.py)
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.801 | 0.777 | -0.024 | 0.008 | 34 |
| 53 | 0.663 | 0.658 | -0.005 | 0.011 | 32 |
| 1_11 | 0.688 | 0.682 | -0.006 | 0.011 | 32 |
| 1_143 | 0.680 | 0.678 | -0.002 | 0.020 | 26 |
| 1_37 | 0.723 | 0.702 | -0.021 | 0.025 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.711 → this 0.699 (-0.012 ± 0.007)
- inclusive (sensitivity), same thresholds: 0.594 → 0.586 (-0.008)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.817 | 0.805 | -0.012 | 603 |
| untagged | 0.717 | 0.706 | -0.011 | 7485 |
| background | 0.692 | 0.676 | -0.016 | 9412 |
| quiet | 0.231 | 0.228 | -0.003 | 2231 |
| faint | 0.000 | 0.000 | +0.000 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

### a0.50_d8 vs yam64h16 a0.50_d8
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.824 | 0.820 | -0.004 | 0.009 | 34 |
| 53 | 0.657 | 0.654 | -0.003 | 0.008 | 32 |
| 1_11 | 0.678 | 0.677 | -0.001 | 0.016 | 32 |
| 1_143 | 0.715 | 0.701 | -0.014 | 0.016 | 26 |
| 1_37 | 0.760 | 0.738 | -0.022 | 0.013 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.727 → this 0.718 (-0.009 ± 0.006)
- inclusive (sensitivity), same thresholds: 0.613 → 0.601 (-0.012)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.815 | 0.821 | +0.006 | 603 |
| untagged | 0.731 | 0.720 | -0.011 | 7485 |
| background | 0.704 | 0.702 | -0.002 | 9412 |
| quiet | 0.255 | 0.237 | -0.018 | 2231 |
| faint | 0.000 | 0.067 | +0.067 | 35 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Reading: the sparse grid costs -0.012 +- 0.007 (a0.25) and -0.009 +- 0.006 (a0.50) against the full-grid twins, each under 2 SD, and buys 1.15x / 1.11x speed in the same session. Per fold, a0.25's 1_29 (-0.024 +- 0.008) is the only delta beyond 2 SD; 1_37 (-0.021 +- 0.025, a0.25; -0.022 +- 0.013, a0.50) is unsure; the rest are within SD. Tiers move together and small: a0.25 loud -0.012, untagged -0.011, background -0.016; a0.50 loud +0.006, untagged -0.011, background -0.002, quiet -0.018. No tier is singled out (unlike yam64h16lo, whose loss sat in background buzz), consistent with the coarser grid being a uniform small approximation error.
WSD curve: both trunks ran the full 56000 (a0.50 hit@K gains +1.81, +1.02, +0.39 after the first doubling; the stop rule did not cut it).

## Conclusion
Not falsified: a0.25_d8 0.699 (>= 0.690) at 1.15x its twin's same-session speed (>= 1.1x); prediction (0.705 +- 0.02 at ~6.2x) held at 6.03x. Both students are frontier steps on the logged table: a0.25 is the best student at >= 6x (next 0.687, yam64h32 a0.50_d8, which is faster, 6.34x, in the same session), a0.50 the best at >= 4.6x (0.718; next faster is yam64h16 a0.25_d8 0.711 @5.25x). Gains over everything at its speed or faster are +0.012 and +0.007, under the 0.02 repeat bar. Skipping every 2nd DFT bin above ~1 kHz is a cheap, tier-neutral speedup. Per the 2026-10-19 injunction, no follow-up (e.g. sp3, or sp2 on yam64h32) is proposed.

Friction: the chain's mmtz re-export first died on `--fe-impl mmtz` missing from export_student.py's choices (mmt-overlap's code never reached main); fixed in this branch (dfd0a2e), reported via loop_signal friction.
