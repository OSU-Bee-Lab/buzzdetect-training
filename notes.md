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
Both students, rung C, seed 1, WSD 56000 (stop rule: a0.50_d8 hit@K +0.58/+0.71, a0.25_d8 +1.47/+1.01), `_mmt` rows:

| front end | frames/patch | a0.25_d8 | a0.50_d8 |
|---|---|---|---|
| yam64h16 (25 ms win, m=2) | 60 | 0.711 @ 4.57x | 0.727 @ 3.77x |
| **yam64h20 (25 ms win, m=2)** | 48 | **0.693 @ 4.82x** | **0.706 @ 4.13x** |
| yam64h24 (24 ms win = hop, m=1) | 40 | 0.658 @ 6.69x | 0.689 @ 5.59x |
| yam64h32 (32 ms win = hop, m=1) | 30 | 0.658 / 0.676 (s1/s2) @ ~6.9x | 0.687 @ 6.06x |

WSD curves: a0.25_d8 0.662 / 0.674 / 0.687 / 0.693 at 7k-56k (still climbing slowly); a0.50_d8
0.695 / 0.701 / 0.706 / 0.706 (flat past 28k).

$ tools/results.py fe_C_yam64h24_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt fe_C_yam64h20_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.752 | 0.783 | +0.031 | 0.022 | 34 |
| 53 | 0.647 | 0.653 | +0.006 | 0.014 | 32 |
| 1_11 | 0.595 | 0.647 | +0.052 | 0.016 | 32 |
| 1_143 | 0.600 | 0.660 | +0.060 | 0.018 | 26 |
| 1_37 | 0.696 | 0.723 | +0.027 | 0.015 | 15 |
- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.658 → this 0.693 (+0.035 ± 0.008)
- inclusive (sensitivity), same thresholds: 0.552 → 0.579 (+0.027)
- folds that missed fpr 0.005: none
| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.797 | 0.809 | +0.012 | 603 |
| untagged | 0.668 | 0.698 | +0.030 | 7485 |
| background | 0.649 | 0.678 | +0.029 | 9412 |
| quiet | 0.207 | 0.227 | +0.020 | 2231 |
| faint | 0.122 | 0.067 | -0.055 | 35 |
± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.
$ tools/results.py fe_C_yam64h16_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt fe_C_yam64h20_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.801 | 0.783 | -0.018 | 0.009 | 34 |
| 53 | 0.663 | 0.653 | -0.010 | 0.012 | 32 |
| 1_11 | 0.688 | 0.647 | -0.041 | 0.013 | 32 |
| 1_143 | 0.680 | 0.660 | -0.020 | 0.016 | 26 |
| 1_37 | 0.723 | 0.723 | +0.000 | 0.021 | 15 |
- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.711 → this 0.693 (-0.018 ± 0.007)
- inclusive (sensitivity), same thresholds: 0.594 → 0.579 (-0.015)
- folds that missed fpr 0.005: none
| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.817 | 0.809 | -0.008 | 603 |
| untagged | 0.717 | 0.698 | -0.019 | 7485 |
| background | 0.692 | 0.678 | -0.014 | 9412 |
| quiet | 0.231 | 0.227 | -0.004 | 2231 |
| faint | 0.000 | 0.067 | +0.067 | 35 |
± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.
$ tools/results.py fe_C_yam64h24_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt fe_C_yam64h20_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.806 | 0.825 | +0.019 | 0.011 | 34 |
| 53 | 0.655 | 0.646 | -0.009 | 0.016 | 32 |
| 1_11 | 0.634 | 0.649 | +0.015 | 0.012 | 32 |
| 1_143 | 0.632 | 0.664 | +0.032 | 0.022 | 26 |
| 1_37 | 0.718 | 0.747 | +0.029 | 0.015 | 15 |
- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.689 → this 0.706 (+0.017 ± 0.007)
- inclusive (sensitivity), same thresholds: 0.577 → 0.592 (+0.015)
- folds that missed fpr 0.005: none
| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.812 | 0.813 | +0.001 | 603 |
| untagged | 0.686 | 0.707 | +0.021 | 7485 |
| background | 0.698 | 0.700 | +0.002 | 9412 |
| quiet | 0.216 | 0.232 | +0.016 | 2231 |
| faint | 0.067 | 0.067 | +0.000 | 35 |
± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.
$ tools/results.py fe_C_yam64h16_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt fe_C_yam64h20_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.824 | 0.825 | +0.001 | 0.010 | 34 |
| 53 | 0.657 | 0.646 | -0.011 | 0.012 | 32 |
| 1_11 | 0.678 | 0.649 | -0.029 | 0.017 | 32 |
| 1_143 | 0.715 | 0.664 | -0.051 | 0.015 | 26 |
| 1_37 | 0.760 | 0.747 | -0.013 | 0.013 | 15 |
- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.727 → this 0.706 (-0.021 ± 0.006)
- inclusive (sensitivity), same thresholds: 0.613 → 0.592 (-0.021)
- folds that missed fpr 0.005: none
| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.815 | 0.813 | -0.002 | 603 |
| untagged | 0.731 | 0.707 | -0.024 | 7485 |
| background | 0.704 | 0.700 | -0.004 | 9412 |
| quiet | 0.255 | 0.232 | -0.023 | 2231 |
| faint | 0.000 | 0.067 | +0.067 | 35 |
± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.
$ tools/results.py fe_C_twofast32_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt fe_C_yam64h20_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.807 | 0.783 | -0.024 | 0.014 | 34 |
| 53 | 0.674 | 0.653 | -0.021 | 0.014 | 32 |
| 1_11 | 0.628 | 0.647 | +0.019 | 0.029 | 32 |
| 1_143 | 0.673 | 0.660 | -0.013 | 0.021 | 26 |
| 1_37 | 0.725 | 0.723 | -0.002 | 0.016 | 15 |
- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.701 → this 0.693 (-0.008 ± 0.009)
- inclusive (sensitivity), same thresholds: 0.590 → 0.579 (-0.011)
- folds that missed fpr 0.005: none
| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.810 | 0.809 | -0.001 | 603 |
| untagged | 0.705 | 0.698 | -0.007 | 7485 |
| background | 0.701 | 0.678 | -0.023 | 9412 |
| quiet | 0.233 | 0.227 | -0.006 | 2231 |
| faint | 0.000 | 0.067 | +0.067 | 35 |
± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

**Reading.** Sensitivity: h20 sits between its twins, as predicted. a0.25_d8 is +0.035 +- 0.008 over h24
(every fold up, 1_11 and 1_143 by ~3 SD; untagged/background +0.03, loud +0.012) and -0.018 +- 0.007 under h16
(1_11 -0.041 +- 0.013, the rest within ~1-2 SD; untagged -0.019, background -0.014). a0.50_d8 is +0.017 +- 0.007
over h24 and -0.021 +- 0.006 under h16 (1_143 -0.051 +- 0.015; untagged and quiet -0.02). So per frame lost the cost is
roughly linear from 60 to 48 frames (~0.0015-0.0018/frame) and steeper from 48 to 40 (~0.002-0.004/frame), with the
h24 step also changing the window/overlap (24 ms, no overlap) -- the two are confounded there.

Speed: the surprise. h20 is only 1.05x (a0.25_d8) and 1.10x (a0.50_d8) faster than h16 despite 20% fewer frames,
while h24 and h32 (window = hop) are ~1.4x faster than h20, and h32 (30 frames) is no faster than h24 (40). Speed
tracks the mmt front end's `m = ceil(window / hop)` (MatMuls summed over shifted row slices: h16/h20 m=2, h24/h32
m=1) far more than the trunk's frame count. Single timing per model; the x YAMNet 20 s column agrees (h20 4.17 vs
h16 4.01 vs h24 4.30).

Frontier: neither is on it. a0.25_d8 (0.693 @ 4.82x) is dominated by twofast32:a0.50_d8 (0.701 @ 5.33x;
-0.008 +- 0.009, a tie in headline at 1.1x the speed); a0.50_d8 (0.706 @ 4.13x) is dominated by yam64h16:a0.25_d8
(0.711 @ 4.57x). Repeat rule: no gain > 0.02 over everything at its speed or faster, no repeat due.

## Conclusion
Prediction held for sensitivity (a0.25_d8 0.693 vs 0.685 +- 0.015; a0.50_d8 0.706 vs 0.705 +- 0.015); falsifier
(a0.25_d8 <= 0.668) not met: the cliff is not complete at 48 frames, cost ~linear 60->48 and steeper 48->40. But the
speed half of the hypothesis failed: hop 20 buys only 1.05-1.10x over hop 16, so h20 is dominated on both widths and
adds no frontier point. Tiers: losses vs h16 are untagged/background (quiet on a0.50_d8), loud flat.
The lead it leaves: the speed split is m (overlapping-window MatMuls in the mmt front end), not frames, so either
the m=2 graph has removable overhead (an export-only fix, like matmul-stft) or a window = hop front end at 48-60
frames is the cheap route. Followed up as the next experiment (mmt overlap cost, prefix_bench on h20 vs h24).
